import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import ffmpeg


def create_video_importer_gui():
    root = tk.Tk()
    root.title("极致视频处理工具 - v2.1")
    root.geometry("800x600")
    root.minsize(700, 500)

    # 设置主题样式（跨平台兼容）
    style = ttk.Style()
    style.configure('TButton', padding=6, relief='flat', background="#4CAF50")
    style.configure('TLabel', font=('Segoe UI', 10))
    style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))

    # 数据存储
    video_files = []
    output_path = [None]
    temp_dir = "temp"
    total_frames = [0]
    merged_video_path = os.path.join(temp_dir, "merged_output.mp4")
    split_mode = False

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    filetypes = (
        ("视频文件", "*.mp4 *.mov *.avi *.mkv *.flv *.wmv *.webm"),
        ("所有文件", "*.*")
    )

    def select_videos():
        files = filedialog.askopenfilenames(title="选择视频文件", filetypes=filetypes)
        if files:
            video_files.clear()
            video_files.extend(files)
            update_listbox()

    def update_listbox():
        listbox.delete(0, tk.END)
        for f in video_files:
            listbox.insert(tk.END, os.path.basename(f))

    def select_output_folder():
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            output_path[0] = folder
            output_label.config(text=f"输出路径：{folder}")

    def analyze_and_concatenate():
        nonlocal split_mode
        split_mode = False

        if len(video_files) < 2:
            messagebox.showwarning("提示", "请至少选择两个视频进行拼接。")
            return

        cap1 = cv2.VideoCapture(video_files[0])
        w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        ext1 = os.path.splitext(video_files[0])[1].lower()
        cap1.release()

        for path in video_files[1:]:
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ext = os.path.splitext(path)[1].lower()
            cap.release()

            if w != w1 or h != h1:
                messagebox.showerror("错误", "分辨率不同，请选择相同分辨率的视频。")
                return

            if abs(fps - fps1) > 0.01:
                messagebox.showerror("错误", "帧率不同，请选择相同帧率的视频。")
                return

            if ext != ext1:
                messagebox.showerror("错误", "格式不同，请选择相同格式的视频。")
                return

        if not output_path[0]:
            messagebox.showwarning("提示", "请先选择输出文件夹。")
            return

        inputs = [ffmpeg.input(file) for file in video_files]
        joined = ffmpeg.concat(*inputs, v=1, a=1).node
        output = ffmpeg.output(joined[0], joined[1], merged_video_path)

        try:
            ffmpeg.run(output, overwrite_output=True)  # 添加 overwrite_output=True 参数
            messagebox.showinfo("成功", f"视频已保存到：\n{merged_video_path}")
            
            cap = cv2.VideoCapture(merged_video_path)
            total_frames[0] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            frame_count_label.config(text=f"最大可裁切数量：{total_frames[0]}")
            split_mode = True
            
        except Exception as e:
            messagebox.showerror("错误", f"视频拼接失败：{str(e)}")

    def get_frame_transform_func(format_type):
        def transform(frame):
            if format_type == "gray":
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif format_type == "hsv":
                return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            elif format_type == "gray_hsv":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif format_type == "gray_png":
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:  # png
                return frame
        return transform

    def process_frame_range(args):
        start, end, filename, format_type, folder, indices = args
        cap = cv2.VideoCapture(filename)
        transform = get_frame_transform_func(format_type)
        results = []

        for idx in range(start, min(end, len(indices))):
            success, frame = cap.read()
            if not success:
                break
            transformed = transform(frame)
            img_name = f"{indices[idx]:07d}.png" if format_type.endswith("png") else f"{indices[idx]:07d}.jpg"
            img_path = os.path.join(folder, img_name)
            cv2.imwrite(img_path, transformed)
            results.append(idx)
        cap.release()
        return results

    def split_video_to_images():
        nonlocal split_mode
        if not split_mode:
            messagebox.showwarning("提示", "请先点击【确定】按钮加载视频信息。")
            return

        try:
            count = int(split_entry.get())
            if count <= 0:
                raise ValueError
        except:
            messagebox.showerror("错误", "请输入有效的正整数作为分割数量。")
            return

        if total_frames[0] < count:
            messagebox.showerror("错误", "分割数量不能大于视频总帧数。")
            return

        cores = int(cpu_combo.get())

        format_type = format_var.get()
        shuffle_flag = shuffle_var.get()

        cap = cv2.VideoCapture(merged_video_path)
        interval = total_frames[0] // count
        indices = list(range(count))
        if shuffle_flag:
            np.random.shuffle(indices)

        frame_indices = []
        frame_idx = 0
        saved_idx = 0
        while saved_idx < count and cap.grab():
            if frame_idx % interval == 0:
                frame_indices.append(frame_idx)
                saved_idx += 1
            frame_idx += 1
        cap.release()

        frame_ranges = distribute_frames_evenly(len(frame_indices), cores)
        tasks = []

        for start, end in frame_ranges:
            tasks.append((
                start,
                end,
                merged_video_path,
                format_type,
                output_path[0],  # 使用用户选择的输出路径
                indices[start:end]
            ))

        with ThreadPoolExecutor(max_workers=cores) as executor:
            futures = []
            for task in tasks:
                future = executor.submit(process_frame_range, task)
                futures.append(future)

            for future in futures:
                future.result()

        messagebox.showinfo("完成", f"已保存 {count} 张图片到 {os.path.abspath(output_path[0])} 文件夹。")

    def distribute_frames_evenly(length, num_workers):
        avg = length // num_workers
        remainder = length % num_workers
        ranges = []
        start = 0
        for i in range(num_workers):
            end = start + avg + (1 if i < remainder else 0)
            ranges.append((start, end))
            start = end
        return ranges

    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 视频操作区
    video_frame = ttk.LabelFrame(main_frame, text="视频操作", padding=10)
    video_frame.pack(fill=tk.X, pady=10)

    ttk.Button(video_frame, text="选择视频文件", width=20, command=select_videos).pack(side=tk.LEFT, padx=5)
    ttk.Button(video_frame, text="选择输出路径", width=20, command=select_output_folder).pack(side=tk.LEFT, padx=5)
    ttk.Button(video_frame, text="确定", width=10, command=analyze_and_concatenate).pack(side=tk.LEFT, padx=5)

    output_label = ttk.Label(main_frame, text="输出路径：未选择", anchor="w")
    output_label.pack(fill=tk.X, pady=5)

    # 参数设置区
    param_frame = ttk.LabelFrame(main_frame, text="参数设置", padding=10)
    param_frame.pack(fill=tk.X, pady=10)

    grid_frame = ttk.Frame(param_frame)
    grid_frame.pack()

    ttk.Label(grid_frame, text="最大可裁切数量：").grid(row=0, column=0, sticky=tk.W, pady=5)
    frame_count_label = ttk.Label(grid_frame, text="0")
    frame_count_label.grid(row=0, column=1, sticky=tk.W, padx=10)

    ttk.Label(grid_frame, text="目标图片数量：").grid(row=1, column=0, sticky=tk.W, pady=5)
    split_entry = ttk.Entry(grid_frame, width=10)
    split_entry.grid(row=1, column=1, sticky=tk.W, padx=10)

    ttk.Label(grid_frame, text="输出格式：").grid(row=2, column=0, sticky=tk.W, pady=5)
    format_var = tk.StringVar(value="png")
    format_menu = ttk.OptionMenu(grid_frame, format_var, "png", "png", "hsv", "gray", "gray_png", "gray_hsv")
    format_menu.grid(row=2, column=1, sticky=tk.W, padx=10)

    ttk.Label(grid_frame, text="CPU 核心数：").grid(row=3, column=0, sticky=tk.W, pady=5)
    cpu_values = [str(i) for i in range(1, os.cpu_count() + 1)]
    cpu_combo = ttk.Combobox(grid_frame, values=cpu_values, width=8)
    cpu_combo.current(len(cpu_values) - 1)  # 默认选最后一个（最大核心数）
    cpu_combo.grid(row=3, column=1, sticky=tk.W, padx=10)

    shuffle_var = tk.BooleanVar()
    ttk.Checkbutton(grid_frame, text="打乱图片顺序", variable=shuffle_var).grid(row=4, column=0, sticky=tk.W, pady=5)

    # 开始按钮
    ttk.Button(main_frame, text="开始", width=20, command=split_video_to_images).pack(pady=15)

    # 列表框
    listbox = tk.Listbox(main_frame, height=8)
    listbox.pack(fill=tk.BOTH, expand=True, pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_video_importer_gui()



