import open3d as o3d    
import numpy as np
import torch
from plyfile import PlyData
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from diff_gaussian_rasterization import GaussianRasterizer



if torch.cuda.is_available():
    # 可以设置为 'cuda'（默认GPU）或 'cuda:0'（指定第一个GPU）
    torch.set_default_device('cuda')  
    print("默认设备已设置为GPU")
else:
    torch.set_default_device('cpu')
    print("CUDA不可用，默认设备设置为CPU")



def PC_show(file_name):
    # 加载PLY点云
    pcd = o3d.io.read_point_cloud(file_name)
    print(pcd)
    print(np.asarray(pcd.points).shape)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

    # 体素下采样
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([downpcd])

    # 重新计算法线
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))


class GaussianRenderer:
    def __init__(self, ply_path):
        # 初始化OpenGL窗口
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(1024, 768)
        glutCreateWindow(b"3D Gaussian Splatting Viewer")
        
        # 加载高斯数据
        self.gaussian_data = self.load_ply(ply_path)
        
        # 初始化相机参数
        self.camera_pos = np.array([0, 0, 3], dtype=np.float32)
        self.camera_target = np.array([0, 0, 0], dtype=np.float32)
        self.camera_up = np.array([0, 1, 0], dtype=np.float32)
        self.fov = 45.0
        
        # 交互状态
        self.mouse_x, self.mouse_y = 0, 0
        self.rotate_x, self.rotate_y = 0, 0
        self.scale = 1.0
        self.translate_x, self.translate_y = 0, 0
        self.is_dragging = False
        
        # 设置回调函数
        glutDisplayFunc(self.render)
        glutReshapeFunc(self.resize)
        glutMouseFunc(self.mouse_button)
        glutMotionFunc(self.mouse_move)
        glutKeyboardFunc(self.keyboard)
        glutIdleFunc(self.idle)
        
        # 初始化光栅化器
        self.renderer = GaussianRasterizer(
            raster_settings={
                'image_height': 768,
                'image_width': 1024,
                'tanfovx': np.tan(np.radians(self.fov/2)),
                'tanfovy': np.tan(np.radians(self.fov/2)),
                'bg': torch.tensor([0, 0, 0], dtype=torch.float32).cuda(),
                'scale_modifier': 1.0,
                'sh_degree': 3,
                'prefiltered': False
            }
        )
    
    def load_ply(self, file_path):
        """加载PLY文件并转换为PyTorch张量"""
        plydata = PlyData.read(file_path)
        data = plydata.elements[0].data
        
        xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
        opacities = np.asarray(data['opacity'])[..., np.newaxis]
        features_dc = np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=1)
        rotations = np.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']], axis=1)
        scales = np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1)
        
        return {
            'xyz': torch.tensor(xyz, dtype=torch.float32).cuda(),
            'opacities': torch.tensor(opacities, dtype=torch.float32).cuda(),
            'features_dc': torch.tensor(features_dc, dtype=torch.float32).cuda(),
            'rotations': torch.tensor(rotations, dtype=torch.float32).cuda(),
            'scales': torch.tensor(scales, dtype=torch.float32).cuda()
        }
    
    def get_view_matrix(self):
        """计算当前视角矩阵"""
        view = np.eye(4)
        view = self.translate_matrix(view, self.translate_x, self.translate_y, 0)
        view = self.rotate_matrix(view, self.rotate_x, 1, 0, 0)
        view = self.rotate_matrix(view, self.rotate_y, 0, 1, 0)
        view = self.scale_matrix(view, self.scale)
        return view
    
    def translate_matrix(self, m, x, y, z):
        """平移矩阵"""
        m[3, 0] += x
        m[3, 1] += y
        m[3, 2] += z
        return m
    
    def rotate_matrix(self, m, angle, x, y, z):
        """旋转矩阵"""
        c = np.cos(np.radians(angle))
        s = np.sin(np.radians(angle))
        t = 1 - c
        
        # 构建旋转矩阵
        rm = np.array([
            [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y, 0],
            [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x, 0],
            [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c,   0],
            [0,            0,            0,           1]
        ])
        return np.dot(m, rm)
    
    def scale_matrix(self, m, s):
        """缩放矩阵"""
        m[0, 0] *= s
        m[1, 1] *= s
        m[2, 2] *= s
        return m
    
    def render(self):
        """渲染场景"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置投影矩阵
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, 1024/768, 0.1, 100.0)
        
        # 设置模型视图矩阵
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        view_matrix = self.get_view_matrix()
        glMultMatrixf(view_matrix.T)
        
        # 使用CUDA光栅化器渲染
        rendered_image = self.render_gaussians(view_matrix)
        
        # 将渲染结果显示到OpenGL窗口
        glDrawPixels(1024, 768, GL_RGB, GL_FLOAT, rendered_image.cpu().numpy())
        glutSwapBuffers()
    
    def render_gaussians(self, view_matrix):
        """使用CUDA光栅化器渲染高斯泼溅"""
        # 计算相机位置
        campos = np.linalg.inv(view_matrix)[:3, 3]
        
        # 设置渲染参数
        self.renderer.raster_settings['viewmatrix'] = torch.tensor(view_matrix, dtype=torch.float32).cuda()
        self.renderer.raster_settings['projmatrix'] = self.gluPerspectiveMatrix(self.fov, 1024/768, 0.1, 100.0)
        self.renderer.raster_settings['campos'] = torch.tensor(campos, dtype=torch.float32).cuda()
        
        # 执行渲染
        rendered_image, _ = self.renderer(
            means3D=self.gaussian_data['xyz'],
            means2D=torch.zeros_like(self.gaussian_data['xyz'][:, :2]),
            shs=None,
            colors_precomp=None,
            opacities=self.gaussian_data['opacities'],
            scales=self.gaussian_data['scales'],
            rotations=self.gaussian_data['rotations'],
            cov3D_precomp=None
        )
        return rendered_image
    
    def resize(self, width, height):
        """窗口大小调整回调"""
        glViewport(0, 0, width, height)
        glutPostRedisplay()
    
    def mouse_button(self, button, state, x, y):
        """鼠标按钮回调"""
        if button == GLUT_LEFT_BUTTON:
            self.is_dragging = (state == GLUT_DOWN)
            self.mouse_x, self.mouse_y = x, y
        elif button == 3:  # 滚轮上滚
            self.scale *= 1.1
        elif button == 4:  # 滚轮下滚
            self.scale *= 0.9
        glutPostRedisplay()
    
    def mouse_move(self, x, y):
        """鼠标移动回调"""
        if self.is_dragging:
            dx = x - self.mouse_x
            dy = y - self.mouse_y
            self.rotate_y += dx * 0.5
            self.rotate_x += dy * 0.5
            self.mouse_x, self.mouse_y = x, y
            glutPostRedisplay()
    
    def keyboard(self, key, x, y):
        """键盘回调"""
        if key == b'w':
            self.translate_y += 0.1
        elif key == b's':
            self.translate_y -= 0.1
        elif key == b'a':
            self.translate_x -= 0.1
        elif key == b'd':
            self.translate_x += 0.1
        glutPostRedisplay()
    
    def idle(self):
        """空闲回调"""
        glutPostRedisplay()

    @staticmethod
    def gluPerspectiveMatrix(fov, aspect, near, far):
        """生成GLU透视投影矩阵"""
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        return np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description='一个处理文件的示例脚本。')
    parser.add_argument('-f', '--file',  # `-f` 是短选项，`--file` 是完整选项
                        type=str,        
                        required=True,
                        help='需要可视化的.ply文件的路径')
    parser.add_argument('-g', '--gaussian',  # 是否为gaussian文件
                        action='store_true',     
                        help='渲染GS文件时启用，否则默认启用点云渲染')
    args = parser.parse_args()


    if args.gaussian:
        renderer = GaussianRenderer(args.file)
        glutMainLoop()
    else:
        PC_show(args.file)
