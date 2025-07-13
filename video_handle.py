import os
import subprocess
import multiprocessing
import tempfile
from tkinter import (
    Tk, Label, Button, Frame, StringVar, Spinbox, IntVar, Entry,
    messagebox, filedialog, OptionMenu
)

# ffmpeg 路径，报错自己改
FFMPEG_PATH = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
FFPROBE_PATH = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffprobe.exe")
################################################################################
MAX_THREADS = multiprocessing.cpu_count()
def detect_nvidia_gpus():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            return [f"GPU {idx}: {name}" for idx, name in
                    (line.split(',') for line in result.stdout.strip().splitlines())]
    except Exception:
        pass
    return []
def detect_intel_qsv():
    try:
        result = subprocess.run(
            ['powershell', '-Command',
             'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'],
            capture_output=True, text=True, timeout=3)
        return any("Intel" in line for line in result.stdout.splitlines())
    except Exception:
        return False

def detect_amd_vaapi():
    return False

def get_video_duration(path):
    try:
        out = subprocess.run(
            [FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True)
        return float(out.stdout.strip())
    except:
        return 0.0

def get_video_fps(path):
    try:
        out = subprocess.run(
            [FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True)
        num, den = out.stdout.strip().split('/')
        return float(num) / float(den)
    except:
        return 30.0

def get_video_frame_count(path):
    try:
        res = subprocess.run(
            [FFPROBE_PATH, '-v', 'error', '-count_frames',
             '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_read_frames',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True)
        if res.stdout.strip().isdigit():
            return int(res.stdout.strip())
        res2 = subprocess.run(
            [FFPROBE_PATH, '-v', 'error',
             '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_frames',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True)
        if res2.stdout.strip().isdigit():
            return int(res2.stdout.strip())
    except:
        pass

    return int(get_video_duration(path) * get_video_fps(path))

def merge_videos(paths, tmpdir):
    listf = os.path.join(tmpdir, 'list.txt')
    with open(listf, 'w', encoding='utf-8') as f:
        for p in paths:
            safe = p.replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    merged = os.path.join(tmpdir, 'merged.mp4')
    cmd = [FFMPEG_PATH, '-noautorotate',
           '-hide_banner', '-loglevel', 'error',
           '-probesize', '5M', '-analyzeduration', '100M',
           '-f', 'concat', '-safe', '0', '-i', listf,
           '-map', '0:v:0', '-c', 'copy', '-threads', str(MAX_THREADS),
           merged, '-y']
    subprocess.run(cmd, check=True)
    return merged

def build_ffmpeg_cmd(path, out_folder, indices, hw, gpu):
    out_pattern = os.path.join(out_folder, '%07d.png')
    cmd = [FFMPEG_PATH, '-hide_banner', '-loglevel', 'error']

    if hw == 'NVIDIA CUDA' and gpu:
        idx = int(gpu.split()[1].strip(':'))
        cmd += ['-hwaccel', 'cuda', '-hwaccel_device', str(idx), '-c:v', 'h264_cuvid']
    elif hw == 'Intel QSV':
        cmd += ['-hwaccel', 'qsv', '-c:v', 'h264_qsv']
    cmd += ['-probesize', '5M', '-analyzeduration', '100M',
            '-threads', str(MAX_THREADS), '-i', path,
            '-map', '0:v', '-an', '-sn']


    expr = '+'.join(f"eq(n\\,{i})" for i in indices)
    vf_chain = f"select='{expr}'"

    cmd += ['-vf', vf_chain, '-vsync', 'vfr', out_pattern, '-y']
    return cmd

def extract_from_merged(merged, out, total, hw, gpu):
    frames = get_video_frame_count(merged)
    if frames <= 0:
        raise RuntimeError('无法获取帧数')
    step = frames / (total + 1)
    idxs = [int((i + 1) * step) for i in range(total)]
    cmd = build_ffmpeg_cmd(merged, out, idxs, hw, gpu)
    subprocess.run(cmd, check=True)

def extract_by_total(paths, out, total, hw, gpu, workers):
    if total <= 0 or not paths:
        return
    with tempfile.TemporaryDirectory() as td:
        merged = merge_videos(paths, td)
        extract_from_merged(merged, out, total, hw, gpu)

###########################################################################
#下面是ui

def main():
    root = Tk()
    root.title("视频帧提取工具（支持硬件加速）")
    root.geometry("600x600")
    frame = Frame(root)
    frame.pack(pady=10)

    selected_files = StringVar()
    out_dir = StringVar()
    total_images = IntVar(value=100)
    core_num = IntVar(value=MAX_THREADS)
    hw_var = StringVar(value='CPU')
    gpu_var = StringVar()
    max_label = StringVar(value='最大可提取帧数: 0')

    gpus = detect_nvidia_gpus()
    if gpus:
        gpu_var.set(gpus[0])

    def update_max():
        files = selected_files.get().split(';') if selected_files.get() else []
        count = sum(get_video_frame_count(p) for p in files if os.path.exists(p))
        max_label.set(f"最大可提取帧数: {count}")

    def on_start():
        files = selected_files.get().split(';') if selected_files.get() else []
        out = out_dir.get()
        num = total_images.get()
        hw = hw_var.get()
        gpu = gpu_var.get() if hw == 'NVIDIA CUDA' else None
        if not files or not out:
            messagebox.showerror('错误', '请选择视频和输出目录')
            return
        if num <= 0:
            messagebox.showerror('错误', '请输入大于0的图片总数')
            return
        status.config(text='处理中...', fg='blue')
        root.update()
        try:
            extract_by_total(files, out, num, hw, gpu, core_num.get())
            status.config(text='完成', fg='green')
        except Exception as e:
            status.config(text=f'失败: {e}', fg='red')
        finally:
            root.update()

    def select_files():
        paths = filedialog.askopenfilenames(filetypes=[('视频文件','*.mp4;*.mov;*.avi')])
        if paths:
            selected_files.set(';'.join(paths))
            update_max()

    def select_out():
        d = filedialog.askdirectory()
        if d:
            out_dir.set(d)

    row = 0
    Label(frame, text='选择视频文件:', width=20, anchor='e').grid(row=row, column=0, pady=4)
    Button(frame, text='浏览', width=15, command=select_files).grid(row=row, column=1)
    row += 1
    Label(frame, textvariable=selected_files, wraplength=450).grid(row=row, column=0, columnspan=2)
    row += 1
    Label(frame, text='输出目录:', width=20, anchor='e').grid(row=row, column=0, pady=4)
    Button(frame, text='选择', width=15, command=select_out).grid(row=row, column=1)
    row += 1
    Label(frame, textvariable=out_dir).grid(row=row, column=0, columnspan=2)
    row += 1
    Label(frame, text='总共想提取的图片数:', width=20, anchor='e').grid(row=row, column=0, pady=4)
    Entry(frame, textvariable=total_images, width=20).grid(row=row, column=1)
    row += 1
    Label(frame, text='使用的 CPU 核心数:', width=20, anchor='e').grid(row=row, column=0, pady=4)
    Spinbox(frame, from_=1, to=MAX_THREADS, textvariable=core_num, width=18).grid(row=row, column=1)
    row += 1
    hw_opts = ['CPU']
    if detect_intel_qsv(): hw_opts.append('Intel QSV')
    if detect_amd_vaapi(): hw_opts.append('AMD VAAPI')
    if gpus: hw_opts.append('NVIDIA CUDA')
    Label(frame, text='加速方式:', width=20, anchor='e').grid(row=row, column=0, pady=4)
    OptionMenu(frame, hw_var, *hw_opts).grid(row=row, column=1)
    row += 1
    Label(frame, text='选择 GPU 名称:', width=20, anchor='e').grid(row=row, column=0, pady=4)
    gpu_menu = OptionMenu(frame, gpu_var, *gpus)
    gpu_menu.grid(row=row, column=1)
    if not gpus: gpu_menu.config(state='disabled')
    row += 1
    Label(frame, textvariable=max_label, fg='gray').grid(row=row, column=0, columnspan=2)
    row += 1
    Button(frame, text='开始提取帧', width=20, command=on_start).grid(row=row, column=0, columnspan=2, pady=10)
    row += 1
    status = Label(frame, text='', fg='blue')
    status.grid(row=row, column=0, columnspan=2)

    root.mainloop()

if __name__ == '__main__':
    main()