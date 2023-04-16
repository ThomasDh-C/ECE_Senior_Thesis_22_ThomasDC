# ECE Senior Thesis Fall 2022 - Thomas Dhome-Casanova

## Directory layout
- *helper_funcs*
    - all_funcs_to_excel - read all of the nvdlahw repo and extract all documentation possible on parameters for each register:
        - all_param_shifts_for_csb.json - used by converters for converting compiler IR to MMIO sequence for the simulators
        - all_regs.xlsx - all registers without parameter annotations
        - all_regs_with_params.xlsx - all registers with automated parameter annotations
        - all_regs_with_params_2.xlsx - all registers with automated and manual parameter annotations for missed registers
    - resnet_20 - a series of ResNet-20 models that could not be succesfully converted to a format accepted by Glenside
    - sdp_funcs_to_json - all_funcs_to_excel = similar to above but only for sdp
- *hlscnn_conv2d_example* - an example from a previous work on offloading conv2d operators to the HLS-CNN accelerator
- *sdp_driver_python_translation* - a translation of NVIDIA's sdp driver from C to Python. Used for interim report. None of this code ended up being used for the final D2A integration as a different approach was needed for integrating all sub-units into D2A.
- *test_thesis* - 2 files to test the D2A integration
    - resnet18.py - run TVM's testing ResNet-18 model on the NVDLA using direct matching
    - unit_tests.py - run unit tests on all simulatable sub-units of the NVDLA

## Extending this work
As simulators become available for the CDP, PDP and RUBIK unit tests can be added to test_thesis/unit_tests.py. 

It will be necessary to uncomment the necessary direct matches in ilanvdla.py.

Next, the file resnet18.py can be used to validate that offloading has been done correctly.