import os
import tensorrt as trt

# Create a logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def initialize_builder(use_fp16=False, workspace_size=(1 << 31)):  # 2GB expressed using bit shift
    """
    Initialize and configure the builder for TensorRT.

    Args:
        use_fp16 (bool): Use FP16 if supported and requested.
        workspace_size (int): Maximum workspace size for the builder.

    Returns:
        Tuple[trt.Builder, trt.BuilderConfig]: Returns the builder and builder configuration.
    """
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)
    config.max_workspace_size = workspace_size  # 2GB using bit shift

    if builder.platform_has_fast_fp16 and use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    return builder, config

def parse_onnx_model_static(builder, onnx_file_path, batch_size=2):
    """
    Parse an ONNX model and create a network in TensorRT with fixed batch size.

    Args:
        builder (trt.Builder): TensorRT builder.
        onnx_file_path (str): Path to the ONNX model file.
        batch_size (int): Fixed batch size to set for all inputs.

    Returns:
        trt.INetworkDefinition: Returns the TensorRT network definition.
    """
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Read and parse the ONNX model
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('❌ Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("✅ Completed parsing ONNX file")
    
    # Set fixed batch size for all inputs
    for i in range(network.num_inputs):
        shape = list(network.get_input(i).shape)
        shape[0] = batch_size
        network.get_input(i).shape = shape

    return network

def build_and_save_engine(builder, network, config, engine_file_path):
    """
    Build and save the TensorRT engine.

    Args:
        builder (trt.Builder): TensorRT builder.
        network (trt.INetworkDefinition): TensorRT network.
        config (trt.BuilderConfig): Builder configuration.
        engine_file_path (str): Path to save the serialized engine.
    """
    # Remove existing engine file if it exists
    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception as e:
            print(f"Cannot remove existing file: {engine_file_path}. Error: {e}")

    print("Creating TensorRT Engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"===> Serialized Engine Saved at: {engine_file_path}")
    else:
        print("❌ Failed to build engine")

# Main function for fixed batch size
def main_fixed():
    batch_size = 1
    onnx_file_path = "data/weights/model_384x640.onnx"
    engine_file_path = "data/weights/model_384x640_FP16.trt"

    # Initialize builder and configuration
    builder, config = initialize_builder(use_fp16=True)

    # Parse ONNX and get network
    network = parse_onnx_model_static(builder, onnx_file_path, batch_size=batch_size)

    # Build and save engine if parsing succeeded
    if network:
        build_and_save_engine(builder, network, config, engine_file_path)

# Entry point
if __name__ == "__main__":
    main_fixed()
