from .node import DepthSplattingModelLoader, DepthSplattingNode, InpaintingInferenceNode

NODE_CLASS_MAPPINGS = {
    "DepthSplattingModelLoader": DepthSplattingModelLoader,
    "DepthSplattingNode": DepthSplattingNode,
    "InpaintingInferenceNode": InpaintingInferenceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthSplattingModelLoader": "Depth Splatting Model Loader",
    "DepthSplattingNode": "Depth Splatting Node",
    "InpaintingInferenceNode": "Inpainting Inference Node",
}
