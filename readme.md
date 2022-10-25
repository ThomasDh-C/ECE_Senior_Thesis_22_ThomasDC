# ECE Senior Thesis Fall 2022 - Thomas Dhome-Casanova

## Normal systems layout
PyTorch --> TVM Relay --> Compiler IR pattern that decides what runs on CPU vs accelerator
Compiler IR pattern matches TVM Relay to IR functions. These IR functions map directly to instructions on the accelerator (ILA)

## Programs
- *match_conv2d.py*: if run with empty data and test folder then using relay api manually creates 2 basic 1 layer conv models (1 that doesn't (mod_wo_acc) and 1 that does uses the ILACNN accelerator (mod)) and runs them. The final TVMscript represenations are saved in the test folder. The mod model simulates the ILACNN with SystemC. This dumps the input and all weights (only a kernel in this case) matrix in the NHWC format as a JSON file. The input, weight and conv_result are also saved without any structure e.g. 3 rows/ cols as txt files. The structured input and weight array are found in test/inputs.log. The ILA instructions are found in the test folder ... assume systemC made this?