import torch
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from model import Transformer 
from dataclasses import dataclass
from typing import List

import torch_mlir
from torch_mlir.dynamo import make_simple_dynamo_backend
import torch._dynamo as dynamo
from torch.fx.experimental.proxy_tensor import make_fx
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from shark.shark_inference import SharkInference
from io import BytesIO


torch.manual_seed(0)

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024

def compile_via_shark(model, inputs):
    x =  model(*inputs)
    return x

torch.set_grad_enabled(False)
inputs = (torch.zeros(1, 1).to(torch.int64), 0)
model = Transformer(params=ModelArgs).eval()
x = model(*inputs)
print(x)


def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
    removed_indexes = []
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert (
                len(node.args) == 1
            ), "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, (list, tuple)):
                node_arg = list(node_arg)
                node_args_len = len(node_arg)
                for i in range(node_args_len):
                    curr_index = node_args_len - (i + 1)
                    if node_arg[curr_index] is None:
                        removed_indexes.append(curr_index)
                        node_arg.pop(curr_index)
                node.args = (tuple(node_arg),)
                break

    if len(removed_indexes) > 0:
        fx_g.graph.lint()
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
    removed_indexes.sort()
    return removed_indexes


def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    """
    Replace tuple with tuple element in functions that return one-element tuples.
    Returns true if an unwrapping took place, and false otherwise.
    """
    unwrapped_tuple = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert (
                len(node.args) == 1
            ), "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    unwrapped_tuple = True
                    break

    if unwrapped_tuple:
        fx_g.graph.lint()
        fx_g.recompile()
    return unwrapped_tuple


def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert (
                len(node.args) == 1
            ), "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                return len(node_arg) == 0
    return False


def transform_fx(fx_g):
    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            if node.target in [
                torch.ops.aten.empty,
            ]:
                # aten.empty should be filled with zeros.
                if node.target in [torch.ops.aten.empty]:
                    with fx_g.graph.inserting_after(node):
                        new_node = fx_g.graph.call_function(
                            torch.ops.aten.zero_,
                            args=(node,),
                        )
                        node.append(new_node)
                        node.replace_all_uses_with(new_node)
                        new_node.args = (node,)

    fx_g.graph.lint()


@make_simple_dynamo_backend
def refbackend_torchdynamo_backend(
    fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    # handling usage of empty tensor without initializing
    transform_fx(fx_graph)
    fx_graph.recompile()
    if _returns_nothing(fx_graph):
        return fx_graph
    removed_none_indexes = _remove_nones(fx_graph)
    was_unwrapped = _unwrap_single_tuple_return(fx_graph)

    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors"
    )
    mlir_module.dump()

    bytecode_stream = BytesIO()
    mlir_module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    shark_module = SharkInference(
        mlir_module=bytecode, device="cpu", mlir_dialect="tm_tensor"
    )
    shark_module.compile()

    def compiled_callable(*inputs):
        inputs = [x.numpy() for x in inputs]
        result = shark_module("forward", inputs)
        if was_unwrapped:
            result = [
                result,
            ]
        if not isinstance(result, list):
            result = torch.from_numpy(result)
        else:
            result = tuple(torch.from_numpy(x) for x in result)
            result = list(result)
            for removed_index in removed_none_indexes:
                result.insert(removed_index, None)
            result = tuple(result)
        return result

    return compiled_callable

# torch._dynamo.config.suppress_errors = True

dynamo_callable = dynamo.optimize(refbackend_torchdynamo_backend)(compile_via_shark)

x = dynamo_callable(model, inputs)
print(x)

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
# model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").eval()


# @torch.inference_mode()
# def shark_module(model, input):
    # return model(input)


# sequence = "Hey I am doing just right"

# tokenized_inputs = tokenizer(sequence, return_tensors="pt")
# print(tokenized_inputs["input_ids"].shape)
# print(tokenized_inputs["input_ids"].dtype)

# # model(tokenized_inputs["input_ids"])
# inputs = tokenized_inputs["input_ids"]


# dynamo_callable = dynamo.optimize(refbackend_torchdynamo_backend)(shark_module)

# x = dynamo_callable(model, inputs)
# print(x)
