import os
from StereoCrafter.depth_splatting_inference import DepthCrafterDemo, ForwardWarpStereo
import torch
import numpy as np
from torchvision import write_video
from decord import VideoReader, cpu


class DepthSplattingModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pre_trained_path": ("STRING", {}),
                "unet_path": ("STRING", {}),
            }
        }

    CATEGORY = "depth_splatting"
    FUNCTION = "load_models"
    RETURN_TYPES = ("DEPTH_SPLATTING_MODEL",)

    def load_models(self, pre_trained_path, unet_path):
        model = DepthCrafterDemo(unet_path=unet_path, pre_trained_path=pre_trained_path)
        return (model,)


class DepthSplattingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DEPTH_SPLATTING_MODEL", {}),
                "input_video_path": ("STRING", {}),
                "output_video_path": ("STRING", {}),
                "max_disp": ("FLOAT", {"default": 20.0}),
                "process_length": ("INT", {"default": -1}),
            }
        }

    CATEGORY = "depth_splatting"
    FUNCTION = "infer"
    RETURN_TYPES = ("STRING",)

    def infer(
        self, model, input_video_path, output_video_path, max_disp, process_length
    ):
        video_depth, depth_vis = model.infer(
            input_video_path,
            output_video_path,
            process_length,
        )

        vid_reader = VideoReader(input_video_path, ctx=cpu(0))
        original_fps = vid_reader.get_avg_fps()
        input_frames = vid_reader[:].asnumpy() / 255.0

        if process_length != -1 and process_length < len(input_frames):
            input_frames = input_frames[:process_length]

        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

        left_video = (
            torch.tensor(input_frames).permute(0, 3, 1, 2).float().contiguous().cuda()
        )
        disp_map = torch.tensor(video_depth).unsqueeze(1).float().contiguous().cuda()

        disp_map = disp_map * 2.0 - 1.0
        disp_map = disp_map * max_disp

        right_video, occlusion_mask = stereo_projector(left_video, disp_map)

        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = (
            occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)
        )

        video_grid_top = np.concatenate([input_frames, depth_vis], axis=2)
        video_grid_bottom = np.concatenate([occlusion_mask, right_video], axis=2)
        video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=1)

        write_video(
            output_video_path,
            video_grid * 255.0,
            fps=original_fps,
            video_codec="h264",
            options={"crf": "16"},
        )

        return (output_video_path,)


class InpaintingInferenceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pre_trained_path": ("STRING", {}),
                "unet_path": ("STRING", {}),
                "input_video_path": ("STRING", {}),
                "save_dir": ("STRING", {}),
            }
        }

    CATEGORY = "inpainting"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)

    def main(self, pre_trained_path, unet_path, input_video_path, save_dir):
        model_loader = DepthSplattingModelLoader()
        model = model_loader.main(pre_trained_path, unet_path)[0]

        output_video_path = os.path.join(save_dir, "camel_splatting_results.mp4")
        node = DepthSplattingNode()
        status = node.main(model, input_video_path, output_video_path, 20.0, -1)[0]
        return (status,)
