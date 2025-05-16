图片合成器
概述
图片合成器是一款基于 Python 和 Tkinter 开发的 GUI 应用程序，允许用户在照片上添加章图（印章），并进行调整（如缩放、旋转、透明度、饱和度等），最终保存为 PNG 或 JPEG 格式的图像。
功能

照片导入：支持导入 JPG、JPEG 和 PNG 格式的照片。
章图添加：支持添加 JPG、JPEG 和 PNG 格式的章图（印章）。
章图调整：
拖动和缩放章图。
调整亮度、透明度、饱和度和旋转角度。
支持多种混合模式（正片叠底、正常、叠加、柔光）。


照片旋转：支持将照片顺时针旋转 90 度。
保存功能：将编辑后的图像保存为 PNG 或 JPEG 格式。
界面优化：主界面在打开照片后自动隐藏，关闭编辑窗口后重新显示。

环境要求

Python 3.6 或更高版本。
Windows 操作系统（打包后的 .exe 文件适用于 Windows）。

安装依赖
1. 安装依赖（使用国内镜像源）
本程序依赖 Pillow 和 numpy 库，使用以下一条命令通过国内镜像源安装：
pip install Pillow numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

2. 安装 PyInstaller（使用国内镜像源）
PyInstaller 用于将 Python 脚本打包为 .exe 文件，使用以下命令安装：
pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple

打包命令
将程序打包为一个单独的 .exe 文件，使用以下命令：
pyinstaller -F -w -i D:\imagecomposer\pic.ico D:\imagecomposer\image_composer.py

打包命令说明：

-F：打包为单个 .exe 文件。
-w：运行时隐藏控制台窗口。
-i D:\imagecomposer\pic.ico：指定图标文件（请确保路径正确）。
D:\imagecomposer\image_composer.py：目标脚本路径（请根据实际路径调整）。

打包后文件：

可执行文件位于 dist\image_composer.exe。

使用方法
1. 运行程序

直接运行 Python 脚本：python image_composer.py


运行打包后的 .exe 文件：双击 dist\image_composer.exe。

2. 操作步骤

打开照片：
点击主界面上的“打开照片”按钮，选择一张图片（支持 JPG、JPEG、PNG 格式）。
主界面会自动隐藏，弹出编辑窗口。


旋转照片（可选）：
点击“旋转 90 度”按钮，将照片顺时针旋转 90 度。


添加章图：
点击“导入章图”按钮，选择一个章图图片（支持 JPG、JPEG、PNG 格式）。


调整章图：
拖动：点击并拖动章图，调整位置。
缩放：拖动章图边缘的控制点（蓝色小方块），调整大小。
删除：右键点击章图，选择“删除章图”。
属性调整：点击“调整”按钮，弹出调整窗口，可以修改：
缩放比例（10-500%）
亮度（0-200%）
透明度（0-100%）
饱和度（0-400%）
旋转角度（0-359°）
混合模式（正片叠底、正常、叠加、柔光）




保存图片：
点击“保存”按钮，选择保存路径和格式（PNG 或 JPEG），保存编辑后的图片。


关闭编辑窗口：
点击窗口关闭按钮，主界面会重新显示。



注意事项

图标文件：打包时需确保 pic.ico 文件存在于指定路径。
性能：对于超大图像，拖动或调整可能有轻微延迟，可尝试调整代码中的 update_interval 参数。
混合模式：红色印章建议使用“叠加”模式以保持鲜艳度。

问题反馈
如遇到任何问题，请提供具体错误信息或反馈，我会尽快协助解决！
