"""Two-source, paired-image mask editor with independent output and joint undo."""
from pathlib import Path
import json
import datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image

from mask_editor_app import MaskEditorApp, normalize_u8
from annotation_modes import MODES
from fusion_storage import save_integrated, load_integrated

PROJECT = Path(__file__).resolve().parent
SIDES = ('before', 'after')


def connected_region(mask, x, y):
    """8-connected foreground containing this exact pixel; background is a no-op."""
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or not mask[y, x]:
        return np.zeros(mask.shape, bool)
    flood = mask.astype(np.uint8).copy()
    cv2.floodFill(flood, None, (int(x), int(y)), 2, flags=8)
    return flood == 2


def merge_region(target, source, bounds, replace=False):
    x0, y0, x1, y1 = bounds
    result = target.copy()
    if replace:
        result[y0:y1, x0:x1] = source[y0:y1, x0:x1]
    else:
        result[y0:y1, x0:x1] |= source[y0:y1, x0:x1]
    return result


class FusionEditor(MaskEditorApp):
    def __init__(self, root, project_root=PROJECT):
        self.project_root = Path(project_root).resolve()
        self.fusion_tool = tk.StringVar(root, value='add')
        self.erase_scope = tk.StringVar(root, value='单侧（当前编辑侧）')
        self.source_choice = tk.StringVar(root, value='Aedge')
        self.reference_view = tk.StringVar(root, value='两套来源对照')
        self.fusion_note = tk.StringVar(root, value='蓝=Before，红=After；在左两格编辑融合结果，右两格为来源参考。')
        self.source_masks = {}
        self.source_task = None
        self.roi = None
        self.roi_start = None
        self.roi_canvas = None
        self.edit_history = []
        self.edit_redo = []
        super().__init__(root)
        self.index_path.set(str(self.project_root/'annotation_index.csv'))
        self.root.geometry('1720x980')
        self.root.minsize(1240, 760)
        self.root.bind('<Control-y>', lambda _e: self.redo())
        self.root.bind('<Escape>', self.cancel_selection)

    def _build(self):
        super()._build()
        # Registered datasets do not need the old raw-image import workflow.
        # Keep its widgets alive for inherited SAM methods, but use a compact
        # toolbar that exposes only this project's editing operations.
        children = self.root.winfo_children()
        children[0].pack_forget()
        children[1].pack_forget()
        bar = ttk.LabelFrame(self.root, text='SEM Pore Annotation Toolkit · 修改结果保存到本项目', padding=7)
        bar.pack(fill=tk.X, before=self.info_label, padx=8, pady=6)
        workflow = ttk.Frame(bar); workflow.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(workflow, text='新图对配准与候选对齐…', command=self.open_registration_workflow).pack(side=tk.LEFT)
        ttk.Button(workflow, text='查看配准报告', command=self.open_registration_report).pack(side=tk.LEFT, padx=5)
        ttk.Label(workflow, textvariable=self.registration_status, wraplength=1000).pack(side=tk.LEFT, padx=6)
        line = ttk.Frame(bar); line.pack(fill=tk.X, pady=2)
        ttk.Button(line, text='上一组', command=self.prev_item).pack(side=tk.LEFT)
        ttk.Button(line, text='下一组', command=self.next_item).pack(side=tk.LEFT, padx=3)
        ttk.Label(line, textvariable=self.navigation_status).pack(side=tk.LEFT, padx=5)
        ttk.Label(line, text='编辑侧：').pack(side=tk.LEFT)
        side = ttk.Combobox(line, textvariable=self.annotation_side, values=SIDES, state='readonly', width=7)
        side.pack(side=tk.LEFT); side.bind('<<ComboboxSelected>>', self.change_annotation_mode)
        for text, tool in [('画笔添加','add'), ('画笔擦除','erase'), ('点击删除整块','delete'), ('框选融合区域','roi')]:
            ttk.Radiobutton(line, text=text, variable=self.fusion_tool, value=tool, command=self.tool_changed).pack(side=tk.LEFT, padx=4)
        ttk.Label(line, text='笔径').pack(side=tk.LEFT)
        ttk.Spinbox(line, from_=1, to=100, textvariable=self.brush_size, width=4).pack(side=tk.LEFT)
        ttk.Button(line, text='撤销 Ctrl+Z', command=self.undo).pack(side=tk.LEFT, padx=5)
        ttk.Button(line, text='重做', command=self.redo).pack(side=tk.LEFT)
        ttk.Button(line, text='保存融合Mask Ctrl+S', command=self.save_mask).pack(side=tk.RIGHT)
        line = ttk.Frame(bar); line.pack(fill=tk.X, pady=3)
        ttk.Label(line, text='擦除／整块删除：').pack(side=tk.LEFT)
        scope = ttk.Combobox(line, textvariable=self.erase_scope, values=('单侧（当前编辑侧）','双侧（同一坐标）'), state='readonly', width=21)
        scope.pack(side=tk.LEFT)
        ttk.Label(line, text='融合来源：').pack(side=tk.LEFT, padx=(12,0))
        ttk.Combobox(line, textvariable=self.source_choice, values=('Aedge','SAM'), state='readonly', width=7).pack(side=tk.LEFT)
        ttk.Button(line, text='局部补充', command=lambda:self.apply_fusion(False)).pack(side=tk.LEFT, padx=4)
        ttk.Button(line, text='局部替换', command=lambda:self.apply_fusion(True)).pack(side=tk.LEFT)
        ttk.Button(line, text='清除框选', command=self.clear_roi).pack(side=tk.LEFT, padx=4)
        ttk.Label(line, text='右两格：').pack(side=tk.LEFT, padx=(10,0))
        view = ttk.Combobox(line, textvariable=self.reference_view, values=('两套来源对照','融合红蓝叠加'), state='readonly', width=15)
        view.pack(side=tk.LEFT); view.bind('<<ComboboxSelected>>', lambda _e:self.refresh_images())
        ttk.Button(line, text='适配窗口', command=self.fit_view).pack(side=tk.LEFT,padx=5)
        ttk.Checkbutton(line, text='SAM交互', variable=self.sam_enabled, command=self.toggle_sam_mode).pack(side=tk.LEFT)
        ttk.Button(line, text='接受SAM', command=self.accept_sam_candidate).pack(side=tk.LEFT)
        ttk.Label(bar, textvariable=self.mask_source_status, foreground='#1769aa').pack(anchor='w',pady=2)
        ttk.Label(bar, textvariable=self.fusion_note, foreground='#9a4e00').pack(anchor='w',pady=2)
        ttk.Label(bar, textvariable=self.sam_status).pack(anchor='w')
        for pane in self.all_panes():
            # The base Before/After panes previously received click only.
            pane.canvas.bind('<B1-Motion>', self.on_drag)
            pane.canvas.bind('<ButtonRelease-1>', self.on_release)
            pane.canvas.bind('<Configure>', lambda _e:self.root.after_idle(self.draw_roi), add='+')

    def all_panes(self):
        return (self.before_pane, self.after_pane, self.mask_pane, self.overlay_pane)

    def open_registration_workflow(self):
        if not self.confirm_discard(): return
        from sem_workbench import RegistrationWindow
        window = tk.Toplevel(self.root)
        def ready(workspace):
            # Editing may have continued while the non-modal registration window
            # was open. Check again before switching projects or output roots.
            if not self.confirm_discard(): return
            self.project_root = Path(workspace).resolve()
            self.rows = []; self.folder_navigation_target = None
            self.index_path.set(str(self.project_root/'annotation_index.csv'))
            self.status_filter.set('ok')
            self.load_index()
            window.destroy()
        RegistrationWindow(window, on_ready=ready)

    def load_index(self):
        session = self.project_root/'new integrate mask/last_session.json'
        if not self.rows and session.is_file():
            saved = json.loads(session.read_text(encoding='utf-8'))
            target = (self.project_root/saved['input_task']).resolve()
            if target.is_relative_to(self.project_root/'ready_to_annotate'):
                self.folder_navigation_target = target
        super().load_index()
        self.root.title(f'当前工作区 | 保存至 new integrate mask | {len(self.rows)} 对')

    def load_current(self):
        previous = self.current_row
        super().load_current()
        if self.current_row is None or self.current_row is not self.rows[self.idx]:
            return
        try:
            integrated = load_integrated(self.project_root,self.current_row,self.mask.shape)
        except Exception as exc:
            messagebox.showerror('独立标签读取失败',str(exc))
            raise
        if integrated:
            masks,saved = integrated
            for side,mask in masks.items():
                self.annotations.layers[side]['mask'] = mask
                self.annotations.layers[side]['original'] = mask.copy()
                self.annotations.used.add(side)
            self.mask = self.annotations.layers[self.active_annotation]['mask']
            self.original_mask = self.annotations.layers[self.active_annotation]['original']
        self.source_masks = {}
        self.source_task = self.annotations.folder.resolve()
        for name in ('Aedge','SAM'):
            self.source_masks[name] = {}
            for side in SIDES:
                path = self.source_task/'source_masks'/name/f'{side}_pore_mask.png'
                mask = None
                if path.is_file():
                    with Image.open(path) as im: mask = np.asarray(im).copy() > 0
                    if mask.shape != self.mask.shape: raise ValueError(f'来源标签尺寸错误：{path}')
                    mask.setflags(write=False)
                self.source_masks[name][side] = mask
        self.roi = self.roi_start = None
        self.edit_history.clear(); self.edit_redo.clear()
        missing = [s for s in SIDES if self.source_masks['SAM'][s] is None]
        self.fusion_note.set('部分 SAM 来源缺失，对应侧初始结果为空；请框选后补充 Aedge 或手工标注。' if missing else
                             '以 SAM 初始化；框选区域后可补充 Aedge 细节，也可用任一来源替换局部。')
        if integrated:
            self.fusion_note.set('已读取 new integrate mask 中的已保存结果，可继续修改。')
        self.refresh_images()

    def update_info(self, include_metrics=True):
        super().update_info(include_metrics)
        if not self.annotations or not self.current_row: return
        self.commit_annotation()
        b,a = (int(np.count_nonzero(self.annotations.layers[s]['mask'])) for s in SIDES)
        available = []
        for name in ('Aedge','SAM'):
            count = sum(self.source_masks.get(name,{}).get(s) is not None for s in SIDES)
            available.append(f'{name}来源 {count}/2')
        self.mask_source_status.set(f'融合结果：蓝 Before {b:,} px / 红 After {a:,} px；'+'；'.join(available)+'；保存到 new integrate mask（按原图命名）')

    def refresh_images(self):
        super().refresh_images()
        if self.before is None or not self.annotations: return
        self.before_pane.title.configure(text='Before 融合结果（蓝色，可编辑）')
        self.after_pane.title.configure(text='After 融合结果（红色，可编辑）')
        if self.reference_view.get() == '两套来源对照':
            side = self.annotation_side.get()
            gray = self.before if side=='before' else self.after
            color = np.array((40,120,255) if side=='before' else (255,60,60))
            for name,pane in [('Aedge',self.mask_pane),('SAM',self.overlay_pane)]:
                rgb = cv2.cvtColor(normalize_u8(gray),cv2.COLOR_GRAY2RGB)
                source = self.source_masks.get(name,{}).get(side) if self.source_task==self.annotations.folder.resolve() else None
                if source is not None:
                    rgb[source] = (rgb[source]*(1-self.alpha.get())+color*self.alpha.get()).astype(np.uint8)
                pane.set_image(rgb)
                pane.title.configure(text=f'{side} · {name} 只读参考（可框选）'+(' · 缺少标签' if source is None else ''))
        self.draw_roi()

    def refresh_mask_pane(self, include_overlay=True):
        self.refresh_images()

    def refresh_panes(self):
        super().refresh_panes(); self.draw_roi()

    def redraw_crop_window(self):
        for pane in self.all_panes(): pane.clear_crop_window()

    def tool_changed(self):
        self.stop_sam_mode('手工编辑；需要SAM时重新开启')
        self.drawing = False; self.last_brush_point = None
        self.mode.set('erase' if self.fusion_tool.get() in ('erase','delete') else 'add')
        hints = {'roi':'在任意图中拖出矩形，再点击局部补充或局部替换；只作用于当前编辑侧。',
                 'delete':'点击左两格删除命中的8邻域连通块；双侧模式分别删除同一坐标命中的块，空白侧不删。',
                 'erase':'拖动画笔擦除；双侧模式在前后图的相同坐标共同擦除。',
                 'add':'在左两格添加融合标签；点击前图编辑蓝色，点击后图编辑红色。'}
        self.fusion_note.set(hints[self.fusion_tool.get()])

    def toggle_sam_mode(self):
        if self.sam_enabled.get() and self.fusion_tool.get() in ('delete','roi'):
            self.fusion_tool.set('add'); self.mode.set('add')
        super().toggle_sam_mode()

    def pick_side(self, pane):
        if pane in (self.before_pane,self.after_pane):
            side = 'before' if pane==self.before_pane else 'after'
            if side!=self.active_annotation:
                self.annotation_side.set(side); self.change_annotation_mode()

    def can_draw_on(self,pane):
        if pane in (self.before_pane,self.after_pane): return True
        return self.reference_view.get()=='融合红蓝叠加' and pane in (self.mask_pane,self.overlay_pane)

    def on_press(self,event):
        pane = self.get_event_pane(event)
        if pane is None or self.mask is None: return
        if self.sam_enabled.get(): return super().on_press(event)
        tool = self.fusion_tool.get()
        if tool=='roi':
            self.pick_side(pane)
            point = pane.canvas_to_image(event.x,event.y)
            if point is None: return
            self.roi_start = point; self.roi_canvas = pane
            self.set_roi(point); return
        if not self.can_draw_on(pane):
            self.fusion_note.set('右侧显示只读来源；请在左侧融合结果中修改，或先框选并执行融合。'); return
        self.pick_side(pane)
        if tool=='delete':
            point = pane.canvas_to_image(event.x,event.y)
            if point is not None: self.delete_component(*point)
            return
        self.mode.set(tool)
        super().on_press(event)

    def on_drag(self,event):
        if self.roi_start is not None:
            pane = self.get_event_pane(event)
            if pane==self.roi_canvas:
                point = pane.canvas_to_image(event.x,event.y)
                if point is not None: self.set_roi(point)
            return
        super().on_drag(event)

    def on_release(self,event):
        if self.roi_start is not None:
            self.roi_start = None; self.roi_canvas = None
            self.fusion_note.set(f'已框选 {self.roi}；选择来源，点击局部补充或局部替换（{self.active_annotation}）。')
            return
        super().on_release(event)

    def set_roi(self,point):
        x,y = point; sx,sy=self.roi_start; h,w=self.mask.shape
        self.roi=(max(0,min(sx,x)),max(0,min(sy,y)),min(w,max(sx,x)+1),min(h,max(sy,y)+1))
        self.draw_roi()

    def draw_roi(self):
        if not hasattr(self,'before_pane'): return
        for pane in self.all_panes():
            pane.canvas.delete('fusion-roi')
            if self.roi and pane.base_pil is not None:
                x0,y0,x1,y1=self.roi
                p0=pane._image_to_canvas(x0,y0);p1=pane._image_to_canvas(x1,y1)
                pane.canvas.create_rectangle(*p0,*p1,outline='#ffff00',width=2,dash=(5,3),tags='fusion-roi')

    def clear_roi(self):
        self.roi=self.roi_start=None;self.roi_canvas=None;self.draw_roi()
        self.fusion_note.set('已清除框选；选择“框选融合区域”后可重新拖出矩形。')

    def cancel_selection(self,event=None):
        self.clear_roi();self.stop_sam_mode('已取消框选及SAM候选')

    def apply_fusion(self,replace=False):
        if not self.roi:
            messagebox.showinfo('请先框选','选择“框选融合区域”，在图像上拖出矩形。');return False
        side=self.active_annotation;name=self.source_choice.get()
        source=self.source_masks.get(name,{}).get(side)
        if source is None:
            messagebox.showinfo('来源标签缺失',f'本组没有 {name} 的 {side} 标签，不能执行融合。');return False
        self.commit_annotation()
        result=merge_region(self.mask,source,self.roi,replace)
        if np.array_equal(result,self.mask):
            self.fusion_note.set('选区内没有需要修改的像素。');return False
        added=int(np.count_nonzero(result & ~self.mask));removed=int(np.count_nonzero(self.mask & ~result))
        self.push_history();self.mask[:]=result;self.annotations.used.add(side)
        self.unsaved=True;self.refresh_images()
        self.fusion_note.set(f'{side} 已用 {name} 局部'+('替换' if replace else '补充')+f'：增加 {added} px，移除 {removed} px；可撤销。')
        return True

    def affected_sides(self):
        return SIDES if self.erase_scope.get().startswith('双侧') else (self.active_annotation,)

    def delete_component(self,x,y):
        self.commit_annotation()
        regions={side:connected_region(self.annotations.layers[side]['mask'],x,y) for side in self.affected_sides()}
        if not any(region.any() for region in regions.values()):
            self.fusion_note.set('点击位置没有标签，未删除任何内容。');return False
        self.push_history()
        for side,region in regions.items():
            self.annotations.layers[side]['mask'][region]=False
            if region.any():self.annotations.used.add(side)
        self.unsaved=True;self.refresh_images()
        self.fusion_note.set('已删除连通块：'+ '；'.join(f'{s} {int(r.sum())} px' for s,r in regions.items())+'；一次撤销恢复整个操作。')
        return True

    def paint(self,ix,iy):
        if self.mode.get()!='erase' or not self.erase_scope.get().startswith('双侧'):
            return super().paint(ix,iy)
        self.commit_annotation()
        diameter=max(1,int(round(self.brush_size.get())));radius=diameter//2
        px,py=self.last_brush_point or (ix,iy);self.last_brush_point=(ix,iy)
        h,w=self.mask.shape
        x0,x1=max(0,min(ix,px)-radius),min(w,max(ix,px)+radius+1)
        y0,y1=max(0,min(iy,py)-radius),min(h,max(iy,py)+radius+1)
        line=np.zeros((y1-y0,x1-x0),np.uint8)
        cv2.line(line,(px-x0,py-y0),(ix-x0,iy-y0),1,1)
        footprint=cv2.dilate(line,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter,diameter)))>0
        for side in SIDES:
            self.annotations.layers[side]['mask'][y0:y1,x0:x1][footprint]=False
            self.annotations.used.add(side)
        self.unsaved=True;self.schedule_mask_refresh()

    def accept_sam_candidate(self):
        side=self._selected_sam_side()
        if side and self.mode.get()=='erase' and self.erase_scope.get().startswith('双侧'):
            candidate=self.sam_candidates[side]
            self.commit_annotation();self.push_history()
            for key in SIDES:
                self.annotations.layers[key]['mask'][candidate]=False;self.annotations.used.add(key)
            self.unsaved=True;self.stop_sam_mode('已用SAM区域双侧擦除，可撤销');self.refresh_images()
        else:
            super().accept_sam_candidate()

    def on_enter_action(self,event=None):
        if self._selected_sam_side() is not None:self.accept_sam_candidate()
        return 'break'

    def snapshot(self):
        self.commit_annotation()
        return {'masks':{k:v['mask'].copy() for k,v in self.annotations.layers.items()},'used':set(self.annotations.used)}

    def push_history(self):
        if self.annotations is None:return
        self.edit_history.append(self.snapshot());self.edit_history=self.edit_history[-30:];self.edit_redo.clear()

    def restore(self,state):
        for key,mask in state['masks'].items():self.annotations.layers[key]['mask']=mask.copy()
        self.annotations.used=set(state['used'])
        self.mask=self.annotations.layers[self.active_annotation]['mask']
        self.unsaved=True;self.drawing=False;self.last_brush_point=None;self.refresh_images()

    def undo(self):
        if not self.edit_history:return
        self.edit_redo.append(self.snapshot());self.restore(self.edit_history.pop())
        self.fusion_note.set('已撤销一步；双侧操作已同时恢复两侧。')

    def redo(self):
        if not self.edit_redo:return
        self.edit_history.append(self.snapshot());self.restore(self.edit_redo.pop())
        self.fusion_note.set('已重做一步。')

    def save_mask(self):
        if not self.annotations:return
        folder=self.annotations.folder.resolve()
        if not folder.is_relative_to(self.project_root/'ready_to_annotate'):
            messagebox.showerror('保存路径错误','融合结果必须保存在 当前工作区 项目内部。');return
        self.commit_annotation()
        state = {'side':self.active_annotation,'selected_source':self.source_choice.get(),
                 'erase_scope':self.erase_scope.get(),'roi':self.roi,
                 'quality':self.quality_status.get(),'note':self.note_text.get()}
        try:
            paths = save_integrated(self.project_root,self.current_row,
                                    {side:self.annotations.layers[side]['mask'] for side in SIDES},state)
        except Exception as exc:
            messagebox.showerror('保存失败',str(exc));return
        self.unsaved=False
        self.update_info()
        self.fusion_note.set('已保存到 new integrate mask；输入标签和来源快照未被覆盖。')
        messagebox.showinfo('融合结果已单独保存',f"目录：{paths['root']}\n\nBefore：{paths['before'].name}\nAfter：{paths['after'].name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Review precomputed Aedge and SAM/fusion masks")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="Prepared workspace containing annotation_index.csv")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if not (workspace / "annotation_index.csv").is_file():
        parser.error("Missing annotation_index.csv; run demo_synthetic.py first.")
    root=tk.Tk()
    try:ttk.Style().theme_use('vista')
    except tk.TclError:pass
    app=FusionEditor(root, workspace)
    root.after(100,app.load_index)
    root.protocol('WM_DELETE_WINDOW',app.close_application)
    root.mainloop()


if __name__=='__main__':main()
