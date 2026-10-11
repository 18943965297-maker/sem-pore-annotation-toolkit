"""Desktop entry point: registration review -> two-source correction."""
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk
from registration_workflow import prepare_pair, comparison_previews, export_workspace

PROJECT = Path(__file__).resolve().parent


def open_editor(workspace):
    return subprocess.Popen([sys.executable, str(PROJECT/'fusion_editor.py'), '--workspace', str(workspace)])


class RegistrationWindow:
    def __init__(self, root, on_ready=open_editor):
        self.root = root
        self.on_ready = on_ready
        self.prepared = None
        self.events = queue.Queue()
        self.busy = False
        self.photos = []
        root.title('SEM 图像配准与双模型辅助标注')
        root.geometry('1120x900'); root.minsize(960, 820)
        self.paths = {k: tk.StringVar(root) for k in ('before', 'after', 'destination')}
        self.mags = {s: tk.StringVar(root, value='自动') for s in ('before', 'after')}
        self.candidates = {f'{source}/{side}': tk.StringVar(root) for source in ('Aedge', 'SAM') for side in ('before', 'after')}
        self.ecc = tk.BooleanVar(root, value=True)
        self.crop = tk.BooleanVar(root, value=True)
        self.reviewed = tk.BooleanVar(root, value=False)
        self.status = tk.StringVar(root, value='选择同一视野的 Before / After 原图。候选标签可选，必须与各自原图尺寸及坐标一致。')
        self.metrics = tk.StringVar(root, value='配准尚未执行')
        for var in [self.paths['before'], self.paths['after']] + list(self.mags.values()) + list(self.candidates.values()) + [self.ecc, self.crop]:
            var.trace_add('write', self.invalidate)
        self.build()
        self.poll_id = root.after(100, self.poll)
        root.bind('<Destroy>', self.cleanup_poll, add='+')

    def build(self):
        frame = ttk.Frame(self.root, padding=14); frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='先对齐，再比较候选与修正标签', font=('Microsoft YaHei UI', 15, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='1 选择图像    →    2 检查配准    →    3 局部融合与人工修正', padding=(0, 4)).pack(anchor='w')
        inputs = ttk.LabelFrame(frame, text='图像与原图坐标中的候选', padding=8); inputs.pack(fill='x', pady=8)
        self.inputs = []
        for row, side in enumerate(('before', 'after')):
            ttk.Label(inputs, text=side.capitalize()+' 原图').grid(row=row, column=0, sticky='w', padx=4, pady=4)
            entry=ttk.Entry(inputs, textvariable=self.paths[side]); entry.grid(row=row, column=1, sticky='ew', padx=4)
            button=ttk.Button(inputs, text='选择图像', command=lambda s=side:self.pick(self.paths[s])); button.grid(row=row, column=2)
            ttk.Label(inputs,text='倍率').grid(row=row,column=3,padx=(10,3))
            mag=ttk.Entry(inputs,textvariable=self.mags[side],width=9);mag.grid(row=row,column=4)
            self.inputs.extend((entry, button, mag))
        for row, (key, var) in enumerate(self.candidates.items(), 2):
            source, side=key.split('/')
            label = 'AP-SAM / 融合' if source=='SAM' else source
            ttk.Label(inputs, text=label+' '+side).grid(row=row, column=0, sticky='w', padx=4, pady=3)
            entry=ttk.Entry(inputs,textvariable=var);entry.grid(row=row,column=1,sticky='ew',padx=4)
            button=ttk.Button(inputs,text='选择二值标签',command=lambda v=var:self.pick(v));button.grid(row=row,column=2,columnspan=3,sticky='w')
            self.inputs.extend((entry,button))
        inputs.columnconfigure(1,weight=1)
        choices=ttk.Frame(frame);choices.pack(fill='x')
        for label, var in [('ECC 精修',self.ecc),('自动裁除 SEM 底部信息栏',self.crop)]:
            control=ttk.Checkbutton(choices,text=label,variable=var);control.pack(side='left',padx=(0,15));self.inputs.append(control)
        self.run_button=ttk.Button(choices,text='执行配准并查看预览',command=self.run);self.run_button.pack(side='right')
        self.progress=ttk.Progressbar(frame,mode='indeterminate');self.progress.pack(fill='x',pady=5)
        ttk.Label(frame,textvariable=self.status,wraplength=1000,foreground='#17466c').pack(anchor='w',pady=3)
        review=ttk.LabelFrame(frame,text='配准检查  绿色为 Before，洋红为 After；稳定纹理应重合',padding=8);review.pack(fill='both',expand=True,pady=6)
        self.preview_labels=[]
        for col, title in enumerate(('配准前叠加（裁剪视野）','配准后叠加（共同有效区）','配准后棋盘格')):
            pane=ttk.Frame(review);pane.grid(row=0,column=col,sticky='nsew',padx=4)
            ttk.Label(pane,text=title).pack()
            label=ttk.Label(pane,text='等待配准',anchor='center');label.pack(fill='both',expand=True,pady=6)
            ttk.Button(pane,text='查看原尺寸',command=lambda i=col:self.show_full_preview(i)).pack(pady=(0,4))
            self.preview_labels.append(label);review.columnconfigure(col,weight=1)
        review.rowconfigure(0,weight=1)
        ttk.Label(frame,textvariable=self.metrics,wraplength=1000).pack(anchor='w',pady=3)
        ttk.Checkbutton(frame,text='我已核对图对、尺度、稳定纹理和有效重叠区；候选与变化类别还需人工检查',variable=self.reviewed,command=self.update_ready).pack(anchor='w',pady=4)
        dest=ttk.Frame(frame);dest.pack(fill='x',pady=5)
        ttk.Label(dest,text='新工作区').pack(side='left')
        entry=ttk.Entry(dest,textvariable=self.paths['destination']);entry.pack(side='left',fill='x',expand=True,padx=8)
        button=ttk.Button(dest,text='选择保存位置',command=self.pick_destination);button.pack(side='left')
        self.inputs.extend((entry,button))
        bottom=ttk.Frame(frame);bottom.pack(fill='x',pady=4)
        self.existing_button=ttk.Button(bottom,text='打开已有融合工作区',command=self.open_existing);self.existing_button.pack(side='left')
        self.finish_button=ttk.Button(bottom,text='创建工作区并开始融合标注',command=self.finish,state='disabled');self.finish_button.pack(side='right')
        self.root.protocol('WM_DELETE_WINDOW',self.close)

    def invalidate(self,*_):
        # Any input change invalidates the preview and the user's approval.
        self.prepared=None;self.reviewed.set(False)
        if hasattr(self,'finish_button'):self.finish_button.configure(state='disabled')
        if hasattr(self,'metrics'):self.metrics.set('当前输入尚未配准；旧预览不能用于本次确认。')

    def show_full_preview(self,index):
        if self.prepared is None:
            self.status.set('请先执行配准，再查看原尺寸预览。');return
        array=list(comparison_previews(self.prepared.result).values())[index]
        window=tk.Toplevel(self.root);window.title('配准预览  原尺寸检查')
        image=Image.fromarray(array)
        canvas=tk.Canvas(window,width=min(image.width,900),height=min(image.height,700),background='#20252b')
        horizontal=ttk.Scrollbar(window,orient='horizontal',command=canvas.xview)
        vertical=ttk.Scrollbar(window,orient='vertical',command=canvas.yview)
        canvas.configure(xscrollcommand=horizontal.set,yscrollcommand=vertical.set)
        canvas.grid(row=0,column=0,sticky='nsew');vertical.grid(row=0,column=1,sticky='ns')
        horizontal.grid(row=1,column=0,sticky='ew')
        window.columnconfigure(0,weight=1);window.rowconfigure(0,weight=1)
        canvas.photo=ImageTk.PhotoImage(image,master=window)
        canvas.create_image(0,0,anchor='nw',image=canvas.photo)
        canvas.configure(scrollregion=(0,0,image.width,image.height))

    def pick(self,var):
        path=filedialog.askopenfilename(parent=self.root,title='选择图像或二值标签',filetypes=[('Images','*.tif *.tiff *.png *.bmp *.jpg *.jpeg'),('All files','*.*')])
        if path:var.set(path)

    def pick_destination(self):
        path=filedialog.asksaveasfilename(parent=self.root,title='选择尚不存在的新工作区目录',initialfile='sem_workspace')
        if path:self.paths['destination'].set(path)

    def set_busy(self,value):
        self.busy=value
        for widget in self.inputs+[self.run_button,self.existing_button]:widget.configure(state='disabled' if value else 'normal')
        if value:self.progress.start(12)
        else:self.progress.stop()
        self.update_ready()

    def run(self):
        if self.busy:return
        before,after=(self.paths[s].get().strip() for s in ('before','after'))
        if not before or not after:
            self.status.set('请先选择 Before 和 After 原图。');return
        self.invalidate();self.set_busy(True);self.status.set('正在配准和对齐候选，请等待。')
        candidates={source:{s:self.candidates[f'{source}/{s}'].get().strip() for s in ('before','after')} for source in ('Aedge','SAM')}
        options=dict(before_magnification=self.mags['before'].get(),after_magnification=self.mags['after'].get(),use_ecc=self.ecc.get(),crop_footer=self.crop.get())
        def work():
            try:self.events.put(('done',prepare_pair(before,after,candidates,**options)))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        self.poll_id = None
        try:
            event,value=self.events.get_nowait()
        except queue.Empty:pass
        else:
            self.set_busy(False)
            if event=='error':self.status.set('配准未完成：'+value)
            else:self.show_prepared(value)
        if self.root.winfo_exists():self.poll_id = self.root.after(100,self.poll)

    def cleanup_poll(self,event):
        if event.widget is self.root and self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None

    def show_prepared(self,prepared):
        self.prepared=prepared;self.reviewed.set(False);self.photos=[]
        for label, array in zip(self.preview_labels,comparison_previews(prepared.result).values()):
            image=Image.fromarray(array);image.thumbnail((300,270))
            photo=ImageTk.PhotoImage(image,master=self.root);self.photos.append(photo);label.configure(image=photo,text='')
        m=prepared.result.metrics
        def num(v):return f'{v:.3f}' if v is not None and __import__('math').isfinite(v) else '不可计算'
        self.metrics.set(f'自动初检：{m.quality}  |  内点 {m.inliers}  |  中位拟合残差 {num(m.reprojection_median_px)} px  |  最终 NCC {num(m.ncc_after)}  |  重叠 {m.overlap_ratio:.1%}\n{m.warning or "请结合稳定纹理复核，真实结构变化不应被强行消除。"}')
        self.status.set('配准完成。预览不代表标签正确；检查后勾选复核项，再创建工作区。')
        self.update_ready()

    def update_ready(self):
        if hasattr(self,'finish_button'):
            ready=self.prepared is not None and self.reviewed.get() and not self.busy
            self.finish_button.configure(state='normal' if ready else 'disabled')

    def finish(self):
        destination=self.paths['destination'].get().strip()
        if not destination:self.status.set('请指定尚不存在的新工作区目录。');return
        try:
            work=export_workspace(self.prepared,destination,reviewed=self.reviewed.get())
        except Exception as exc:self.status.set('未创建标注任务：'+str(exc));return
        self.status.set('已创建工作区并打开融合标注。候选快照保留，编辑结果另行保存。')
        self.invalidate()
        try:self.on_ready(work)
        except Exception as exc:self.status.set('工作区已创建，但编辑器未打开：'+str(exc))

    def open_existing(self):
        path=filedialog.askdirectory(parent=self.root,title='选择包含 annotation_index.csv 的工作区')
        if path:
            if not (Path(path)/'annotation_index.csv').is_file():self.status.set('该目录缺少 annotation_index.csv');return
            self.on_ready(Path(path))

    def close(self):
        if self.busy:
            messagebox.showinfo('配准进行中','请等待当前配准结束后关闭窗口。',parent=self.root);return
        self.root.destroy()


def main():
    root=tk.Tk()
    try:ttk.Style(root).theme_use('vista')
    except tk.TclError:pass
    RegistrationWindow(root);root.mainloop()


if __name__=='__main__':main()
