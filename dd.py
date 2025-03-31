import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from cv_bridge import CvBridge
import torch
import numpy as np
import os
from torchvision import transforms
from inpainting_csu.inpainting_module import Inpainter

class NodeInpainter(Node):
    def __init__(self, weight, model_name, width, height, target_fps):
        super().__init__('inpainting_node')
        self.create_subscription(Image, '/sem/video_frame', self.cb_frame, 10)
        self.create_subscription(Image, '/sem/semantic_frame', self.cb_mask, 10)

        self.pub_restored_frame = self.create_publisher(Image, '/restored_frame', 10)
        
        self.bridge = CvBridge()
        
        self.width, self.height = width, height
        self.resize_transform = transforms.Resize((self.height, self.width))
        self.transform_to_FHD = transforms.Resize((1080, 1920))
    
        self.timer = self.create_timer(1.0/target_fps, self.cb_timer)
        
        self.frame = None
        self.mask = None

        self.inpainter=Inpainter(ckpt=weight,
                                 model_name=model_name,
                                 width=width,height=height)


    def cb_frame(self, msg:Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        tensor_image = torch.from_numpy(cv_image).permute(2, 0, 1)
        tensor_image = tensor_image.float() / 255.0
        self.frame = self.resize_transform(tensor_image)

    def cb_mask(self, msg:Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        tensor_image = torch.from_numpy(cv_image)
        tensor_image = tensor_image.float() 
        tensor_image = tensor_image.unsqueeze(0)
        mask = self.resize_transform(tensor_image)
        self.mask = mask.squeeze()

    def cb_is_masked(self, msg:Bool):
        self.is_masked = msg.data

    def cb_timer(self):
        if self.frame is not None and self.mask is not None:
            self.get_logger().info("restore image")
            restored_image = self.inpainter.process(self.frame, self.mask)
            restored_frame = self.transform_to_FHD(restored_image.squeeze())
            restored_frame = restored_frame.permute(1, 2, 0).cpu().numpy()
            restored_frame = (restored_frame).astype(np.uint8)
            
            frame_msg = self.bridge.cv2_to_imgmsg(restored_frame, encoding="bgr8")
            self.pub_restored_frame.publish(frame_msg)


def main(args=None):
    rclpy.init(args=args)
    workspace_dir = '/ws'

    weight_file = os.path.join(workspace_dir, 'model_weights/DSTT.pth')
    model_name = 'DSTT_432240'
    inpainter = NodeInpainter(weight=weight_file,
                              model_name=model_name,
                              width=432,
                              height=240,
                              target_fps=30)

    rclpy.spin(inpainter)

    inpainter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        
        
