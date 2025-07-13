import os
import subprocess
import multiprocessing
from tkinter import (
    Tk, Label, Button, Frame, StringVar, Spinbox, IntVar, Entry,
    messagebox, filedialog, OptionMenu, CENTER
)
from concurrent.futures import ThreadPoolExecutor

# ffmpeg 路径（需你自行放置对应静态版）
FFMPEG_PATH = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
FFPROBE_PATH = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffprobe.exe")

def detect_nvidia_gpus():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return [f"GPU {line.split(',')[0].strip()}: {line.split(',')[1].strip()}" for line in lines]
    except Exception:
        pass
    return []

def detect_intel_qsv():
    try:
        cmd = ['powershell', '-Command',
               'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        return any("Intel" in line for line in result.stdout.splitlines())
    except Exception:
        return False

def detect_amd_vaapi():
    # 这里简单判断虚拟机或 Linux 可用，Windows 默认 False
    return False

def get_video_duration(video_path):
    try:
        result = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=6000)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration: {e}")
        return 0

def get_video_frame_count(video_path):
    try:
        result = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=6000)
        return int(result.stdout.strip())
    except Exception as e:
        print(f"Error counting frames: {e}")
        # 兜底估算
        duration = get_video_duration(video_path)
        fps = 30
        return int(duration * fps)

def calculate_max_frames(video_paths):
    total_frames = 0
    for path in video_paths:
        if os.path.exists(path):
            total_frames += get_video_frame_count(path)
    return total_frames

# 新增：构建带帧号选择的 FFmpeg 命令
def build_ffmpeg_command_with_indices(video_path, output_folder, frame_indices, hwaccel_type, gpu_name=None):
    output_pattern = os.path.join(output_folder, "%07d.png")  # 防止重名改成了 %05d
    cmd = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error"]

    if hwaccel_type == "NVIDIA CUDA" and gpu_name:
        gpu_index = int(gpu_name.split()[1].replace(':', ''))
        cmd += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_index), "-c:v", "h264_cuvid"]
    elif hwaccel_type == "Intel QSV":
        cmd += ["-hwaccel", "qsv", "-c:v", "h264_qsv"]
    elif hwaccel_type == "AMD VAAPI":
        messagebox.showwarning("提示", "VAAPI 仅支持 Linux，建议放在虚拟机运行！\n当前将回退为 CPU 模式。")
        hwaccel_type = "CPU"

    cmd += ["-i", video_path]

    select_expr = '+'.join(f'eq(n\\,{i})' for i in frame_indices)
    cmd += ["-vf", f"select='{select_expr}'", "-vsync", "vfr", output_pattern, "-y"]
    return cmd

# 新增：按帧索引提取帧
def extract_frames_with_indices(video_path, output_folder, frame_indices, hwaccel, gpu_name):
    cmd = build_ffmpeg_command_with_indices(video_path, output_folder, frame_indices, hwaccel, gpu_name)
    subprocess.run(cmd)

# 替换原函数：按指定总数提取帧（更精确）
def extract_frames_by_total_count(video_paths, output_folder, total_num_images, hwaccel, gpu_name, max_workers):
    all_videos = []
    total_frames = 0

    # 收集所有视频的信息
    for path in video_paths:
        if not os.path.exists(path):
            continue
        frame_count = get_video_frame_count(path)
        total_frames += frame_count
        all_videos.append((path, frame_count))

    if total_frames == 0 or total_num_images <= 0:
        return

    # 计算每个视频应提取哪些帧
    per_video_indices = []

    remaining = total_num_images
    for path, frame_count in all_videos:
        ratio = frame_count / total_frames
        images_to_extract = max(1, round(ratio * total_num_images))
        if images_to_extract > remaining:
            images_to_extract = remaining
        remaining -= images_to_extract

        step = max(1, frame_count // images_to_extract)
        indices = list(range(0, frame_count, step))[:images_to_extract]
        per_video_indices.append((path, indices))

    # 多线程执行
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for path, indices in per_video_indices:
            futures.append(pool.submit(extract_frames_with_indices, path, output_folder, indices, hwaccel, gpu_name))
        for f in futures:
            f.result()

def main():
    root = Tk()
    root.title("视频帧提取工具（支持硬件加速）")
    root.geometry("600x600")
    frame = Frame(root)
    frame.pack(pady=10)

    selected_files = StringVar()
    out_dir = StringVar()
    total_images_var = IntVar(value=100)
    core_var = IntVar(value=multiprocessing.cpu_count())
    hw_var = StringVar(value="CPU")
    gpu_name_var = StringVar()
    max_frame_var = StringVar(value="最大可提取帧数: 0注意！导入视频后卡顿是正常现象，这时候建议去泡杯茶等着")

    gpus = detect_nvidia_gpus()
    if gpus:
        gpu_name_var.set(gpus[0])

    def update_max_frame_label():
        paths = selected_files.get().split(';') if selected_files.get() else []
        count = calculate_max_frames(paths)
        max_frame_var.set(f"最大可提取帧数: {count}注意注意！你的加速方式要和GPU对上号！")

    def on_start():
        files = selected_files.get().split(";") if selected_files.get() else []
        out = out_dir.get()
        total_images = total_images_var.get()
        hw = hw_var.get()
        gpu_name = gpu_name_var.get() if hw == "NVIDIA CUDA" else None
        if not files or not out:
            messagebox.showerror("错误", "请选择视频文件和输出目录！")
            return
        if total_images <= 0:
            messagebox.showerror("错误", "请输入大于0的图片总数！")
            return
        status_label.config(text="处理中，请稍候...", fg="blue")
        root.update()
        try:
            extract_frames_by_total_count(files, out, total_images, hw, gpu_name, core_var.get())
            status_label.config(text="处理完成！", fg="green")
        except Exception as e:
            status_label.config(text=f"处理失败: {str(e)}", fg="red")
        finally:
            root.update()

    def select_files():
        paths = filedialog.askopenfilenames(filetypes=[("视频文件", "*.mp4;*.mov;*.avi")])
        if paths:
            selected_files.set(";".join(paths))
            update_max_frame_label()

    def select_output():
        path = filedialog.askdirectory()
        if path:
            out_dir.set(path)

    # 界面布局
    row = 0
    Label(frame, text="选择视频文件:", anchor="e", width=20).grid(row=row, column=0, sticky="e", pady=4)
    Button(frame, text="浏览", width=15, command=select_files).grid(row=row, column=1, pady=4)
    row += 1
    Label(frame, textvariable=selected_files, wraplength=450, justify=CENTER).grid(row=row, column=0, columnspan=2)
    row += 1

    Label(frame, text="输出目录:", anchor="e", width=20).grid(row=row, column=0, sticky="e", pady=4)
    Button(frame, text="选择", width=15, command=select_output).grid(row=row, column=1, pady=4)
    row += 1
    Label(frame, textvariable=out_dir).grid(row=row, column=0, columnspan=2)
    row += 1

    Label(frame, text="总共想提取的图片数:", anchor="e", width=20).grid(row=row, column=0, sticky="e", pady=4)
    Entry(frame, textvariable=total_images_var, width=20).grid(row=row, column=1)
    row += 1

    Label(frame, text="使用的 CPU 核心数:", anchor="e", width=20).grid(row=row, column=0, sticky="e", pady=4)
    Spinbox(frame, from_=1, to=multiprocessing.cpu_count(), textvariable=core_var, width=18).grid(row=row, column=1)
    row += 1

    hw_options = ["CPU"]
    if detect_intel_qsv():
        hw_options.append("Intel QSV")
    if detect_amd_vaapi():
        hw_options.append("AMD VAAPI")
    if gpus:
        hw_options.append("NVIDIA CUDA")

    Label(frame, text="加速方式:", anchor="e", width=20).grid(row=row, column=0, sticky="e", pady=4)
    OptionMenu(frame, hw_var, *hw_options).grid(row=row, column=1)
    row += 1

    Label(frame, text="选择 GPU 名称:", anchor="e", width=20).grid(row=row, column=0, sticky="e", pady=4)
    gpu_menu = OptionMenu(frame, gpu_name_var, *gpus)
    gpu_menu.grid(row=row, column=1)
    if not gpus:
        gpu_menu.config(state="disabled")
    row += 1

    Label(frame, textvariable=max_frame_var, fg="gray").grid(row=row, column=0, columnspan=2, pady=6)
    row += 1

    Button(frame, text="开始提取帧", width=20, command=on_start).grid(row=row, column=0, columnspan=2, pady=10)
    row += 1

    status_label = Label(frame, text="", fg="blue")
    status_label.grid(row=row, column=0, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    main()