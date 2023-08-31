# Build PITCHNET models

PITCHNET neural network is developped using pytorch.

The model is converted to the ONNX standard: a language
to describe neural network which is technology independent.

Then, the ONNX model is converted to a COREML model that can directly be used in XCode.

# Setup
To compile from ONNX to CoreML, two libraries should be installed with
```shell script
pip3 install --upgrade onnx-coreml
pip3 install --upgrade coremltools
```

## Problem during export
If you get the message
```
Traceback (most recent call last):
  File "./tools/create-minimal-coreml-model.py", line 92, in <module>
    main()
  File "./tools/create-minimal-coreml-model.py", line 83, in main
    onnx_export(model, shape, "SimpleModel.onnx")
  File "./tools/create-minimal-coreml-model.py", line 74, in onnx_export
    output_names=output_names)
  File "/usr/local/lib/python3.7/site-packages/torch/onnx/__init__.py", line 143, in export
    strip_doc_string, dynamic_axes, keep_initializers_as_inputs)
  File "/usr/local/lib/python3.7/site-packages/torch/onnx/utils.py", line 66, in export
    dynamic_axes=dynamic_axes, keep_initializers_as_inputs=keep_initializers_as_inputs)
  File "/usr/local/lib/python3.7/site-packages/torch/onnx/utils.py", line 394, in _export
    operator_export_type, strip_doc_string, val_keep_init_as_ip)
RuntimeError: ONNX export failed: Couldn't export operator aten::log_softmax
```

This means your version of pytorch do not contain a patch from the current
master that allow exporting with log_softmax and softmax whose dimension
is not -1 (through reshaping and transposing input).

Fir you can try to update your PyTorch version, but that may not be enougth.
```shell script
pip3 install torchsummary pytorch-lightning "torchvision>=0.4" "torch>=1.4"
```

You can replace `/torch/onnx/symbolic_opset9.py` by the file
from the master branch to fix this problem (Current commit today is 55c382e).
Here is some command line you may have to adapt to your python installation
(find where the file to replace is located from your error message)
```
wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/onnx/symbolic_opset9.py
cp symbolic_opset9.py /usr/local/lib/python3.7/site-packages/torch/onnx/
```

## Problem during conversion
The master at 1 February 2020 does not contain some
patch required for the export. If you get the error:
```
TypeError: Error while converting op of type: Conv. Error message: provided number axes -1 not supported 
 Please try converting with higher minimum_ios_deployment_target.
You can also provide custom function/layer to convert the model.
```

this means you should add the following patch manually.
From https://github.com/onnx/onnx-coreml/pull/524 we have to change
the file **onnx_coreml/_operators_nd.py** according to:
```python
def _add_conv_like_op(add_func, get_params_func, params_dict,
                      builder, node, graph, err):


+   # To do: Need to avoid dependence on rank for conversion since rank is not always available.
+
   rank = builder._get_rank(node.inputs[0])
+
+   if rank < 0 and node.op_type == "Conv" and "w_shape" in params_dict:
+       rank = len(params_dict["w_shape"])
+
   if rank == 4:
        get_params_func(builder, node, graph, err, params_dict)
```

## Build

Models can be build from the root directory using `./models/build_models.sh`.
