import tkinter as tk
from tkinter import filedialog, Toplevel, Scale, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance
import os
import uuid
import time
import numpy as np
import colorsys

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available, falling back to NumPy implementation.")

class ImageComposer:
    def __init__(self, root):
        self.root = root
        self.root.title("图片合成器")
        self.photo_path = None
        self.photo_image = None
        self.stamps = []
        self.selected_stamp = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.resize_mode = False
        self.last_update = 0
        self.update_interval = 0.1
        self.is_dragging = False
        self.in_drag_mode = False
        self.photo_width = 0
        self.photo_height = 0
        self.original_photo_size = None
        self.scale_factor = 1.0
        self.photo_layer_id = None
        self.stamp_layers = {}
        self.image_cache = {}  # 图像处理缓存
        self.drag_update_timer = None  # 拖动防抖定时器
        self.resize_update_timer = None  # 窗口调整防抖定时器
        self.max_stamps = 10  # 最大图章数量限制

        # Set main window size and center it
        window_width = 250
        window_height = 70
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Main interface
        self.open_button = tk.Button(root, text="打开照片", command=self.open_photo)
        self.open_button.pack(pady=20)

    def open_photo(self):
        self.photo_path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png")])
        if not self.photo_path:
            return

        try:
            self.photo_image = Image.open(self.photo_path).convert("RGBA")
            self.original_photo_size = self.photo_image.size

            self.edit_window = Toplevel(self.root)
            self.edit_window.title("编辑照片")

            screen_width = self.edit_window.winfo_screenwidth()
            screen_height = self.edit_window.winfo_screenheight()
            max_width = int(screen_width * 0.8)
            max_height = int(screen_height * 0.8) - 120

            photo_ratio = self.original_photo_size[0] / self.original_photo_size[1]
            if max_width / max_height > photo_ratio:
                initial_width = int(max_height * photo_ratio)
                initial_height = max_height
            else:
                initial_width = max_width
                initial_height = int(max_width / photo_ratio)

            self.photo_width = initial_width
            self.photo_height = initial_height
            self.scale_factor = self.photo_width / self.original_photo_size[0]

            window_width = initial_width + 40
            window_height = initial_height + 120
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            self.edit_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

            self.canvas = tk.Canvas(self.edit_window, width=self.photo_width, height=self.photo_height, highlightthickness=0)
            self.canvas.pack(pady=10)

            self.update_canvas_photo()

            button_frame = tk.Frame(self.edit_window)
            button_frame.pack(pady=10)
            tk.Button(button_frame, text="旋转 90 度", command=self.rotate_photo_90).pack(side="left", padx=5)
            tk.Button(button_frame, text="导入章图", command=self.import_stamp).pack(side="left", padx=5)
            tk.Button(button_frame, text="调整", command=self.open_adjust_window).pack(side="left", padx=5)
            tk.Button(button_frame, text="保存", command=self.save_image).pack(side="left", padx=5)

            self.canvas.bind("<Button-1>", self.start_drag)
            self.canvas.bind("<B1-Motion>", self.drag)
            self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
            self.canvas.bind("<Button-3>", self.show_context_menu)

            self.edit_window.update_idletasks()
            self.edit_window.bind("<Configure>", self.on_window_resize)

            # Hide the main window after opening the edit window
            self.root.withdraw()

            # Show the main window again when the edit window is closed
            def on_edit_window_close():
                self.edit_window.destroy()
                self.root.deiconify()
                # Reset state
                self.photo_image = None
                self.stamps = []
                self.selected_stamp = None
                self.photo_layer_id = None
                self.stamp_layers.clear()
                self.image_cache.clear()

            self.edit_window.protocol("WM_DELETE_WINDOW", on_edit_window_close)

            self.update_canvas()
        except Exception as e:
            messagebox.showerror("错误", f"无法导入图片: {e}")
            self.root.deiconify()

    def rotate_photo_90(self):
        if not self.photo_image:
            return

        try:
            self.photo_image = self.photo_image.rotate(90, expand=True)
            self.original_photo_size = self.photo_image.size

            screen_width = self.edit_window.winfo_screenwidth()
            screen_height = self.edit_window.winfo_screenheight()
            max_width = int(screen_width * 0.8)
            max_height = int(screen_height * 0.8) - 120

            photo_ratio = self.original_photo_size[0] / self.original_photo_size[1]
            if max_width / max_height > photo_ratio:
                new_width = int(max_height * photo_ratio)
                new_height = max_height
            else:
                new_width = max_width
                new_height = int(max_width / photo_ratio)

            self.photo_width = new_width
            self.photo_height = new_height
            self.scale_factor = self.photo_width / self.original_photo_size[0]

            window_width = new_width + 40
            window_height = new_height + 120
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            self.edit_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

            self.canvas.config(width=self.photo_width, height=self.photo_height)
            self.update_canvas_photo()

            scale_ratio = self.photo_width / self.original_photo_size[0]
            for stamp in self.stamps:
                stamp["x"] = stamp["x"] * scale_ratio
                stamp["y"] = stamp["y"] * scale_ratio
                stamp["last_scale"] = self.scale_factor

            self.update_canvas()
        except Exception as e:
            messagebox.showerror("错误", f"旋转照片时出错: {e}")

    def show_context_menu(self, event):
        if not self.selected_stamp:
            x, y = event.x / self.scale_factor, event.y / self.scale_factor
            for stamp in reversed(self.stamps):
                stamp_img = stamp["image"]
                sx, sy = stamp["x"], stamp["y"]
                scale = stamp["scale"]
                sw, sh = stamp_img.size
                sw_scaled, sh_scaled = int(sw * scale), int(sh * scale)
                if sx - sw_scaled / 2 <= x <= sx + sw_scaled / 2 and sy - sh_scaled / 2 <= y <= sy + sh_scaled / 2:
                    self.selected_stamp = stamp
                    self.update_canvas()
                    break
        
        if self.selected_stamp:
            context_menu = tk.Menu(self.canvas, tearoff=0)
            context_menu.add_command(label="删除章图", command=self.delete_stamp)
            context_menu.post(event.x_root, event.y_root)

    def delete_stamp(self):
        if self.selected_stamp:
            cache_keys = [k for k in self.image_cache.keys() if k[0] == self.selected_stamp["id"]]
            for k in cache_keys:
                del self.image_cache[k]
            self.stamps.remove(self.selected_stamp)
            self.selected_stamp = None
            self.update_canvas()

    def update_canvas_photo(self):
        resized_photo = self.photo_image.resize((self.photo_width, self.photo_height), Image.LANCZOS)
        self.photo_tk = ImageTk.PhotoImage(resized_photo)

        if self.photo_layer_id:
            self.canvas.delete(self.photo_layer_id)

        # 固定底图位置为 (0, 0)，并确保其在最底层
        self.photo_layer_id = self.canvas.create_image(0, 0, image=self.photo_tk, anchor="nw", tags="photo")
        self.canvas.tag_lower(self.photo_layer_id)

    def on_window_resize(self, event):
        if not self.photo_image:
            return

        current_time = time.time()
        if current_time - self.last_update < self.update_interval and not self.is_dragging:
            return
        self.last_update = current_time

        new_width = event.width - 40
        new_height = event.height - 160

        photo_ratio = self.original_photo_size[0] / self.original_photo_size[1]
        if new_width / new_height > photo_ratio:
            new_width = int(new_height * photo_ratio)
        else:
            new_height = int(new_width / photo_ratio)

        if abs(new_width - self.photo_width) > 5 or abs(new_height - self.photo_height) > 5:
            old_scale_factor = self.scale_factor

            self.photo_width = new_width
            self.photo_height = new_height
            self.scale_factor = self.photo_width / self.original_photo_size[0]
            self.canvas.config(width=new_width, height=new_height)

            self.update_canvas_photo()

            scale_ratio = self.scale_factor / old_scale_factor
            for stamp in self.stamps:
                stamp["x"] = stamp["x"] * scale_ratio
                stamp["y"] = stamp["y"] * scale_ratio
                stamp["last_scale"] = self.scale_factor

            self.update_canvas()

    def import_stamp(self):
        if len(self.stamps) >= self.max_stamps:
            messagebox.showwarning("警告", f"图章数量已达上限（{self.max_stamps}个）！")
            return

        stamp_path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png")])
        if not stamp_path:
            return

        try:
            original_img = Image.open(stamp_path)
            if original_img.mode != 'RGBA':
                if original_img.format in ('JPEG', 'JPG'):
                    img_rgba = original_img.convert('RGBA')
                else:
                    img_rgba = original_img.convert('RGBA')
            else:
                img_rgba = original_img

            stamp_id = str(uuid.uuid4())
            self.stamps.append({
                "id": stamp_id,
                "image": img_rgba,
                "x": self.photo_width / self.scale_factor / 2,
                "y": self.photo_height / self.scale_factor / 2,
                "scale": 1.0,
                "brightness": 1.0,
                "opacity": 0.8,
                "saturation": 1.0,
                "rotation": 0.0,
                "blend_mode": "正片叠底",
                "last_scale": self.scale_factor
            })
            self.update_canvas()
        except Exception as e:
            messagebox.showerror("错误", f"无法导入图片: {e}")

    def start_drag(self, event):
        self.selected_stamp = None
        self.resize_mode = False
        self.is_dragging = True
        x, y = event.x / self.scale_factor, event.y / self.scale_factor

        for stamp in reversed(self.stamps):
            stamp_img = stamp["image"]
            sx, sy = stamp["x"], stamp["y"]
            scale = stamp["scale"]
            sw, sh = stamp_img.size
            sw_scaled, sh_scaled = int(sw * scale), int(sh * scale)

            if sx - sw_scaled / 2 <= x <= sx + sw_scaled / 2 and sy - sh_scaled / 2 <= y <= sy + sh_scaled / 2:
                self.selected_stamp = stamp
                self.drag_start_x = x
                self.drag_start_y = y

                border_threshold = 10 / self.scale_factor
                if (abs(x - (sx - sw_scaled / 2)) < border_threshold or
                    abs(x - (sx + sw_scaled / 2)) < border_threshold or
                    abs(y - (sy - sh_scaled / 2)) < border_threshold or
                    abs(y - (sy + sh_scaled / 2)) < border_threshold):
                    self.resize_mode = True
                break

        self.update_canvas()

    def drag(self, event):
        if not self.selected_stamp:
            return

        x, y = event.x / self.scale_factor, event.y / self.scale_factor

        if not self.in_drag_mode:
            self.in_drag_mode = True
            self.selected_stamp["_temp_blend"] = self.selected_stamp.get("blend_mode", "正片叠底")
            self.selected_stamp["blend_mode"] = "正常"

        old_x = self.selected_stamp["x"]
        old_y = self.selected_stamp["y"]

        if self.resize_mode:
            dx = abs(x - old_x)
            dy = abs(y - old_y)
            scale = max(dx, dy) / (self.selected_stamp["image"].size[0] / 2)
            self.selected_stamp["scale"] = max(0.1, min(scale, 5.0))
            self.update_stamp(self.selected_stamp)
        else:
            self.selected_stamp["x"] = x
            self.selected_stamp["y"] = y

            delta_x = (x - old_x) * self.scale_factor
            delta_y = (y - old_y) * self.scale_factor

            layer_id = self.stamp_layers.get(self.selected_stamp["id"])
            if layer_id:
                self.canvas.move(layer_id, delta_x, delta_y)
                self.canvas.move("selection", delta_x, delta_y)

        self.update_selection_rectangle()

        # 防抖更新画布
        if hasattr(self, 'drag_update_timer'):
            self.canvas.after_cancel(self.drag_update_timer)
        self.drag_update_timer = self.canvas.after(100, self.update_canvas)

    def stop_drag(self, event):
        self.is_dragging = False
        if self.in_drag_mode:
            self.in_drag_mode = False
            if self.selected_stamp and "_temp_blend" in self.selected_stamp:
                self.selected_stamp["blend_mode"] = self.selected_stamp["_temp_blend"]
                del self.selected_stamp["_temp_blend"]
            self.update_canvas()

    def multiply_blend(self, base_image, stamp_image, opacity):
        try:
            if base_image.mode != 'RGBA':
                base_image = base_image.convert('RGBA')
            if stamp_image.mode != 'RGBA':
                stamp_image = stamp_image.convert('RGBA')

            # 确保底图区域和图章尺寸匹配
            base_arr = np.array(base_image, dtype=np.float32)
            stamp_arr = np.array(stamp_image.resize(base_image.size, Image.LANCZOS), dtype=np.float32)

            base_rgb = base_arr[:, :, :3]
            stamp_rgb = stamp_arr[:, :, :3]
            stamp_alpha = stamp_arr[:, :, 3:4] / 255.0 * opacity

            multiply = (base_rgb * stamp_rgb) / 255.0

            result_rgb = base_rgb * (1.0 - stamp_alpha) + multiply * stamp_alpha

            result_arr = np.zeros_like(base_arr)
            result_arr[:, :, :3] = np.clip(result_rgb, 0, 255)
            result_arr[:, :, 3] = base_arr[:, :, 3]

            return Image.fromarray(result_arr.astype(np.uint8))
        except Exception as e:
            print(f"Error in multiply blend: {e}")
            return stamp_image

    def apply_opacity(self, image, opacity):
        if opacity >= 1.0:
            return image

        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        r, g, b, a = image.split()
        a = a.point(lambda x: int(x * opacity))
        return Image.merge('RGBA', (r, g, b, a))

    def normal_blend(self, base_image, stamp_image, paste_x, paste_y, opacity=1.0):
        if stamp_image.mode == 'RGBA':
            r, g, b, a = stamp_image.split()
            a = a.point(lambda x: int(x * opacity))
            stamp_image = Image.merge('RGBA', (r, g, b, a))

        result = base_image.copy()
        result.paste(stamp_image, (paste_x, paste_y), stamp_image if stamp_image.mode == 'RGBA' else None)
        return result

    def improved_overlay_blend(self, base_image, stamp_image, paste_x, paste_y, opacity=1.0):
        if base_image.mode != 'RGBA':
            base_image = base_image.convert('RGBA')
        if stamp_image.mode != 'RGBA':
            stamp_image = stamp_image.convert('RGBA')

        result = base_image.copy()
        result_arr = np.array(result, dtype=np.uint8)

        stamp_width, stamp_height = stamp_image.size
        max_x = min(paste_x + stamp_width, base_image.width)
        max_y = min(paste_y + stamp_height, base_image.height)

        stamp_arr = np.array(stamp_image, dtype=np.float32)
        for y in range(paste_y, max_y):
            for x in range(paste_x, max_x):
                if 0 <= x < base_image.width and 0 <= y < base_image.height:
                    sx, sy = x - paste_x, y - paste_y
                    if 0 <= sx < stamp_width and 0 <= sy < stamp_height:
                        base_pixel = result_arr[y, x]
                        stamp_pixel = stamp_arr[sy, sx]

                        r1, g1, b1 = base_pixel[:3]
                        r2, g2, b2, a2 = stamp_pixel

                        if a2 == 0:
                            continue

                        alpha = a2 / 255.0 * opacity
                        if alpha > 0:
                            r = r1 * r2 / 128 if r1 < 128 else 255 - (255 - r1) * (255 - r2) / 128
                            g = g1 * g2 / 128 if g1 < 128 else 255 - (255 - g1) * (255 - g2) / 128
                            b = b1 * b2 / 128 if b1 < 128 else 255 - (255 - b1) * (255 - b2) / 128

                            result_arr[y, x, 0] = int(r1 * (1 - alpha) + r * alpha)
                            result_arr[y, x, 1] = int(g1 * (1 - alpha) + g * alpha)
                            result_arr[y, x, 2] = int(b1 * (1 - alpha) + b * alpha)

        return Image.fromarray(result_arr)

    def improved_soft_light_blend(self, base_image, stamp_image, paste_x, paste_y, opacity=1.0):
        if base_image.mode != 'RGBA':
            base_image = base_image.convert('RGBA')
        if stamp_image.mode != 'RGBA':
            stamp_image = stamp_image.convert('RGBA')

        result = base_image.copy()
        result_arr = np.array(result, dtype=np.uint8)

        stamp_width, stamp_height = stamp_image.size
        max_x = min(paste_x + stamp_width, base_image.width)
        max_y = min(paste_y + stamp_height, base_image.height)

        stamp_arr = np.array(stamp_image, dtype=np.float32)
        for y in range(paste_y, max_y):
            for x in range(paste_x, max_x):
                if 0 <= x < base_image.width and 0 <= y < base_image.height:
                    sx, sy = x - paste_x, y - paste_y
                    if 0 <= sx < stamp_width and 0 <= sy < stamp_height:
                        base_pixel = result_arr[y, x]
                        stamp_pixel = stamp_arr[sy, sx]

                        r1, g1, b1 = base_pixel[:3]
                        r2, g2, b2, a2 = stamp_pixel

                        if a2 == 0:
                            continue

                        alpha = a2 / 255.0 * opacity
                        if alpha > 0:
                            r = r1 * (r2 / 128.0) if r2 < 128 else r1 + (255 - r1) * (r2 - 128) / 128.0
                            g = g1 * (g2 / 128.0) if g2 < 128 else g1 + (255 - g1) * (g2 - 128) / 128.0
                            b = b1 * (b2 / 128.0) if b2 < 128 else b1 + (255 - b1) * (b2 - 128) / 128.0

                            result_arr[y, x, 0] = int(r1 * (1 - alpha) + r * alpha)
                            result_arr[y, x, 1] = int(g1 * (1 - alpha) + g * alpha)
                            result_arr[y, x, 2] = int(b1 * (1 - alpha) + b * alpha)

        return Image.fromarray(result_arr)

    def enhance_saturation(self, image, factor):
        if 0.95 <= factor <= 1.05:
            return image.copy()

        if OPENCV_AVAILABLE:
            try:
                # 转换为 OpenCV 格式 (BGR)
                img_np = np.array(image)
                img_bgr = img_np[:, :, [2, 1, 0, 3]]  # RGBA to BGRA
                img_bgr = img_bgr[:, :, :3]  # Remove alpha channel

                # Convert to HSV
                img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * factor, 0, 255)  # Adjust saturation

                # Convert back to BGR
                img_bgr = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # Restore RGBA
                img_rgba = np.zeros_like(img_np)
                img_rgba[:, :, [2, 1, 0]] = img_bgr
                img_rgba[:, :, 3] = img_np[:, :, 3]  # Restore alpha channel

                return Image.fromarray(img_rgba)
            except Exception as e:
                print(f"Error in enhance_saturation (OpenCV): {e}")
                # Fall back to NumPy implementation
                pass

        # NumPy implementation as fallback
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        r, g, b, a = image.split()

        r_arr = np.array(r, dtype=np.float32) / 255.0
        g_arr = np.array(g, dtype=np.float32) / 255.0
        b_arr = np.array(b, dtype=np.float32) / 255.0

        h_arr = np.zeros_like(r_arr)
        s_arr = np.zeros_like(r_arr)
        v_arr = np.zeros_like(r_arr)

        for i in range(r_arr.shape[0]):
            for j in range(r_arr.shape[1]):
                h, s, v = colorsys.rgb_to_hsv(r_arr[i, j], g_arr[i, j], b_arr[i, j])
                h_arr[i, j] = h
                s_arr[i, j] = s
                v_arr[i, j] = v

        s_arr = np.clip(s_arr * factor, 0, 1)

        for i in range(r_arr.shape[0]):
            for j in range(r_arr.shape[1]):
                r_arr[i, j], g_arr[i, j], b_arr[i, j] = colorsys.hsv_to_rgb(
                    h_arr[i, j], s_arr[i, j], v_arr[i, j]
                )

        r_new = Image.fromarray(np.clip(r_arr * 255, 0, 255).astype(np.uint8))
        g_new = Image.fromarray(np.clip(g_arr * 255, 0, 255).astype(np.uint8))
        b_new = Image.fromarray(np.clip(b_arr * 255, 0, 255).astype(np.uint8))

        return Image.merge('RGBA', (r_new, g_new, b_new, a))

    def process_stamp_image(self, stamp):
        cache_key = (
            stamp["id"],
            stamp["x"],  # 添加位置信息以区分不同图章
            stamp["y"],
            stamp["scale"],
            stamp["opacity"],
            stamp["brightness"],
            stamp["saturation"],
            stamp["rotation"],
            stamp.get("blend_mode", "正片叠底")
        )

        if cache_key in self.image_cache:
            return self.image_cache[cache_key]

        img = stamp["image"]
        scale = stamp["scale"] * self.scale_factor
        opacity = stamp["opacity"]
        brightness = stamp["brightness"]
        saturation = stamp["saturation"]
        rotation = stamp["rotation"]
        blend_mode = stamp.get("blend_mode", "正片叠底")

        processed = img.copy()
        original_size = processed.size  # 保存原始尺寸

        if brightness != 1.0:
            processed = ImageEnhance.Brightness(processed).enhance(brightness)
        if saturation != 1.0:
            processed = self.enhance_saturation(processed, saturation)

        if rotation != 0:
            # Create a transparent canvas to preserve alpha after rotation
            rotated = processed.rotate(rotation, expand=True, resample=Image.BICUBIC)
            new_size = rotated.size
            transparent_canvas = Image.new('RGBA', new_size, (0, 0, 0, 0))
            paste_x = (new_size[0] - processed.size[0]) // 2
            paste_y = (new_size[1] - processed.size[1]) // 2
            transparent_canvas.paste(processed, (paste_x, paste_y), processed)
            processed = transparent_canvas.rotate(rotation, expand=True, resample=Image.BICUBIC)
        else:
            new_size = original_size

        w, h = new_size
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w > 0 and new_h > 0:
            processed = processed.resize((new_w, new_h), Image.LANCZOS)
        else:
            return processed

        if blend_mode == "正片叠底":
            photo_region = self.get_photo_region_under_stamp(stamp, processed.size)
            if photo_region:
                processed = self.multiply_blend(photo_region, processed, opacity)
        else:
            processed = self.apply_opacity(processed, opacity)

        self.image_cache[cache_key] = processed
        return processed

    def get_photo_region_under_stamp(self, stamp, processed_size):
        try:
            x = stamp["x"] * self.scale_factor
            y = stamp["y"] * self.scale_factor

            # 使用旋转后的图章尺寸计算底图区域
            new_w, new_h = processed_size

            left = max(0, int(x - new_w / 2))
            top = max(0, int(y - new_h / 2))
            right = min(self.photo_width, int(x + new_w / 2))
            bottom = min(self.photo_height, int(y + new_h / 2))

            photo_resized = self.photo_image.resize((self.photo_width, self.photo_height), Image.LANCZOS)
            region = photo_resized.crop((left, top, right, bottom))
            return region
        except Exception as e:
            print(f"Error getting photo region: {e}")
            return None

    def draw_stamp(self, stamp):
        try:
            img = stamp["image"]
            x = stamp["x"] * self.scale_factor
            y = stamp["y"] * self.scale_factor

            processed = self.process_stamp_image(stamp)

            if "tk_image" in stamp:
                del stamp["tk_image"]

            tk_img = ImageTk.PhotoImage(processed)
            stamp["tk_image"] = tk_img

            layer_id = self.canvas.create_image(
                x, y,
                image=tk_img,
                anchor="center",
                tags=("stamp", f"stamp_{stamp['id']}")
            )

            self.stamp_layers[stamp["id"]] = layer_id

        except Exception as e:
            print(f"Error drawing stamp: {e}")

    def update_stamp(self, stamp):
        layer_id = self.stamp_layers.get(stamp["id"])
        if layer_id:
            self.canvas.delete(layer_id)

        self.draw_stamp(stamp)

        if self.selected_stamp and self.selected_stamp["id"] == stamp["id"]:
            self.update_selection_rectangle()

    def update_selection_rectangle(self):
        self.canvas.delete("selection")
        if self.selected_stamp:
            sx = self.selected_stamp["x"] * self.scale_factor
            sy = self.selected_stamp["y"] * self.scale_factor
            scale = self.selected_stamp["scale"] * self.scale_factor
            img = self.selected_stamp["image"]
            w, h = img.size
            w_scaled = w * scale
            h_scaled = h * scale

            self.canvas.create_rectangle(
                sx - w_scaled / 2, sy - h_scaled / 2,
                sx + w_scaled / 2, sy + h_scaled / 2,
                outline="blue", width=2, tags="selection"
            )

            control_points = [
                (sx - w_scaled / 2, sy - h_scaled / 2),
                (sx + w_scaled / 2, sy - h_scaled / 2),
                (sx - w_scaled / 2, sy + h_scaled / 2),
                (sx + w_scaled / 2, sy + h_scaled / 2),
            ]

            for point in control_points:
                self.canvas.create_rectangle(
                    point[0] - 4, point[1] - 4,
                    point[0] + 4, point[1] + 4,
                    fill="blue", outline="white", tags="selection"
                )

    def update_canvas(self):
        self.canvas.delete("stamp", "selection")
        for stamp in self.stamps:
            if "tk_image" in stamp:
                del stamp["tk_image"]
        self.stamp_layers.clear()

        # 确保底图始终被重绘
        self.update_canvas_photo()

        for stamp in self.stamps:
            self.draw_stamp(stamp)

        if self.selected_stamp:
            self.update_selection_rectangle()

    def open_adjust_window(self):
        if not self.stamps:
            messagebox.showinfo("提示", "请先导入章图！")
            return

        if hasattr(self, 'adjust_window') and self.adjust_window.winfo_exists():
            self.adjust_window.lift()
            return

        self.adjust_window = Toplevel(self.edit_window)
        self.adjust_window.title("调整章图")
        window_width = 350
        window_height = 550
        screen_width = self.adjust_window.winfo_screenwidth()
        screen_height = self.adjust_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.adjust_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.adjustment_cache = {}

        stamp_ids = [(i, stamp["id"]) for i, stamp in enumerate(self.stamps)]
        self.stamp_combobox = ttk.Combobox(self.adjust_window, state="readonly")
        self.stamp_combobox["values"] = [f"章图 {i+1}" for i, _ in stamp_ids]
        self.stamp_combobox.current(0)

        tk.Label(self.adjust_window, text="选择章图:", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        self.stamp_combobox.pack(pady=(0, 10))

        tk.Label(self.adjust_window, text="缩放比例", font=("Arial", 10, "bold")).pack()
        self.scale_var = tk.DoubleVar(value=self.stamps[0]["scale"] * 100)
        scale_slider = tk.Scale(self.adjust_window, from_=10, to=500, orient="horizontal",
                                variable=self.scale_var, resolution=1)
        scale_slider.pack(fill="x", padx=10)

        tk.Label(self.adjust_window, text="亮度", font=("Arial", 10, "bold")).pack()
        self.brightness_var = tk.DoubleVar(value=self.stamps[0]["brightness"] * 100)
        brightness_scale = tk.Scale(self.adjust_window, from_=0, to=200, orient="horizontal",
                                    variable=self.brightness_var, resolution=1)
        brightness_scale.pack(fill="x", padx=10)

        tk.Label(self.adjust_window, text="透明度", font=("Arial", 10, "bold")).pack()
        self.opacity_var = tk.DoubleVar(value=self.stamps[0]["opacity"] * 100)
        opacity_scale = tk.Scale(self.adjust_window, from_=0, to=100, orient="horizontal",
                                 variable=self.opacity_var, resolution=1)
        opacity_scale.pack(fill="x", padx=10)

        tk.Label(self.adjust_window, text="饱和度", font=("Arial", 10, "bold")).pack()
        self.saturation_var = tk.DoubleVar(value=self.stamps[0]["saturation"] * 100)
        saturation_scale = tk.Scale(self.adjust_window, from_=0, to=400, orient="horizontal",
                                    variable=self.saturation_var, resolution=1)
        saturation_scale.pack(fill="x", padx=10)

        tk.Label(self.adjust_window, text="旋转", font=("Arial", 10, "bold")).pack()
        self.rotation_var = tk.DoubleVar(value=self.stamps[0]["rotation"])
        rotation_scale = tk.Scale(self.adjust_window, from_=0, to=359, orient="horizontal",
                                  variable=self.rotation_var, resolution=1)
        rotation_scale.pack(fill="x", padx=10)

        tk.Label(self.adjust_window, text="混合模式", font=("Arial", 10, "bold")).pack()
        self.blend_mode_var = tk.StringVar()
        blend_modes = ["正片叠底", "正常", "叠加", "柔光"]
        current_idx = self.stamp_combobox.current()
        current_blend = self.stamps[current_idx].get("blend_mode", "正片叠底")
        self.blend_mode_var.set(current_blend)
        blend_menu = ttk.Combobox(self.adjust_window, textvariable=self.blend_mode_var, state="readonly")
        blend_menu["values"] = blend_modes
        blend_menu.pack(fill="x", padx=10, pady=(0, 10))

        self.adjust_debounce_timer = None
        self.active_stamp_index = 0

        def get_selected_stamp_index():
            return self.stamp_combobox.current()

        def on_adjustment_change(*args):
            if self.adjust_debounce_timer:
                self.adjust_window.after_cancel(self.adjust_debounce_timer)
            self.adjust_debounce_timer = self.adjust_window.after(100, self.apply_adjustments)

        def on_stamp_selected(event):
            self.apply_adjustments()
            idx = get_selected_stamp_index()
            self.active_stamp_index = idx
            self.scale_var.set(self.stamps[idx]["scale"] * 100)
            self.brightness_var.set(self.stamps[idx]["brightness"] * 100)
            self.opacity_var.set(self.stamps[idx]["opacity"] * 100)
            self.saturation_var.set(self.stamps[idx]["saturation"] * 100)
            self.rotation_var.set(self.stamps[idx]["rotation"])
            blend_mode = self.stamps[idx].get("blend_mode", "正片叠底")
            self.blend_mode_var.set(blend_mode)
            self.selected_stamp = self.stamps[idx]
            self.update_canvas()

        def on_blend_mode_change(event):
            self.apply_blend_mode(event)

        self.stamp_combobox.bind("<<ComboboxSelected>>", on_stamp_selected)
        blend_menu.bind("<<ComboboxSelected>>", on_blend_mode_change)
        self.scale_var.trace_add("write", on_adjustment_change)
        self.brightness_var.trace_add("write", on_adjustment_change)
        self.opacity_var.trace_add("write", on_adjustment_change)
        self.saturation_var.trace_add("write", on_adjustment_change)
        self.rotation_var.trace_add("write", on_adjustment_change)
        self.blend_mode_var.trace_add("write", on_adjustment_change)

        button_frame = tk.Frame(self.adjust_window)
        button_frame.pack(pady=20, fill="x")
        tk.Button(button_frame, text="应用", command=self.adjust_window.destroy).pack(side="right", padx=10)

    def apply_adjustments(self):
        index = self.stamp_combobox.current()
        if 0 <= index < len(self.stamps):
            stamp = self.stamps[index]

            stamp["scale"] = self.scale_var.get() / 100
            stamp["brightness"] = self.brightness_var.get() / 100
            stamp["opacity"] = self.opacity_var.get() / 100
            stamp["saturation"] = self.saturation_var.get() / 100
            stamp["rotation"] = self.rotation_var.get()

            self.update_canvas()

    def apply_blend_mode(self, event):
        index = self.stamp_combobox.current()
        if 0 <= index < len(self.stamps):
            self.stamps[index]["blend_mode"] = self.blend_mode_var.get()
            self.update_canvas()

    def save_image(self):
        if not self.photo_image:
            messagebox.showerror("错误", "没有打开的照片！")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("JPEG图像", "*.jpg"), ("所有文件", "*.*")]
        )
        if not save_path:
            return

        try:
            base_size = self.original_photo_size
            result = self.photo_image.resize(base_size, Image.LANCZOS)

            for stamp in self.stamps:
                x = stamp["x"] * base_size[0] / (self.photo_width / self.scale_factor)
                y = stamp["y"] * base_size[0] / (self.photo_width / self.scale_factor)

                img = stamp["image"]
                scale = stamp["scale"]
                opacity = stamp["opacity"]
                brightness = stamp["brightness"]
                saturation = stamp["saturation"]
                rotation = stamp["rotation"]
                blend_mode = stamp.get("blend_mode", "正片叠底")

                processed = img.copy()
                original_size = processed.size

                if brightness != 1.0:
                    processed = ImageEnhance.Brightness(processed).enhance(brightness)
                if saturation != 1.0:
                    processed = self.enhance_saturation(processed, saturation)
                if rotation != 0:
                    rotated = processed.rotate(rotation, expand=True, resample=Image.BICUBIC)
                    new_size = rotated.size
                    transparent_canvas = Image.new('RGBA', new_size, (0, 0, 0, 0))
                    paste_x = (new_size[0] - processed.size[0]) // 2
                    paste_y = (new_size[1] - processed.size[1]) // 2
                    transparent_canvas.paste(processed, (paste_x, paste_y), processed)
                    processed = transparent_canvas.rotate(rotation, expand=True, resample=Image.BICUBIC)
                else:
                    new_size = original_size

                w, h = new_size
                new_w = int(w * scale * base_size[0] / self.original_photo_size[0])
                new_h = int(h * scale * base_size[0] / self.original_photo_size[0])

                if new_w > 0 and new_h > 0:
                    processed = processed.resize((new_w, new_h), Image.LANCZOS)

                    paste_x = max(0, int(x - new_w / 2))
                    paste_y = max(0, int(y - new_h / 2))

                    if blend_mode == "正片叠底":
                        temp = Image.new('RGBA', result.size, (0, 0, 0, 0))
                        temp.paste(processed, (paste_x, paste_y), processed)
                        result = self.multiply_blend(result, temp, opacity)
                    elif blend_mode == "叠加":
                        result = self.improved_overlay_blend(result, processed, paste_x, paste_y, opacity)
                    elif blend_mode == "柔光":
                        result = self.improved_soft_light_blend(result, processed, paste_x, paste_y, opacity)
                    else:
                        result = self.normal_blend(result, processed, paste_x, paste_y, opacity)

            if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
                if result.mode == 'RGBA':
                    result = result.convert('RGB')
                result.save(save_path, "JPEG", quality=95)
            else:
                result.save(save_path, "PNG")

            messagebox.showinfo("保存成功", f"图像已保存至:\n{save_path}")

        except Exception as e:
            messagebox.showerror("保存失败", f"保存图像时出错:\n{str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComposer(root)
    root.mainloop()