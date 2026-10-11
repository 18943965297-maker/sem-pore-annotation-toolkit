"""Synthetic capture drift and artificial candidates for the integrated workflow."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from PIL import Image
from registration_workflow import prepare_pair, comparison_previews, export_workspace


def make_capture_demo(destination):
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(23)
    image=rng.normal(135,25,(384,384)).clip(0,255).astype(np.uint8)
    image=cv2.GaussianBlur(image,(5,5),.8)
    for _ in range(90):
        x,y=rng.integers(15,369,2)
        cv2.circle(image,(int(x),int(y)),int(rng.integers(2,8)),int(rng.integers(30,230)),-1)
    yy,xx=np.indices(image.shape)
    large=(xx-150)**2+(yy-180)**2<30**2
    small=(xx-250)**2+(yy-120)**2<8**2
    image[large|small]=35
    transform=cv2.getRotationMatrix2D((192,192),1.2,1.015)
    transform[:,2]+=(7,-5)
    moving=cv2.warpAffine(image,transform,(384,384),borderMode=cv2.BORDER_REFLECT)
    inputs=destination/'inputs';inputs.mkdir()
    Image.fromarray(image).save(inputs/'before.png');Image.fromarray(moving).save(inputs/'after.png')
    candidates={s:{} for s in ('Aedge','SAM')}
    for source,mask in [('SAM',large),('Aedge',large|small)]:
        for side,array in [('before',mask),('after',cv2.warpAffine(mask.astype(np.uint8),transform,(384,384),flags=cv2.INTER_NEAREST)>0)]:
            path=inputs/f'{source}_{side}.png';Image.fromarray(array.astype(np.uint8)).save(path)
            candidates[source][side]=path
    prepared=prepare_pair(inputs/'before.png',inputs/'after.png',candidates,500,500,crop_footer=False)
    for name,array in comparison_previews(prepared.result).items():Image.fromarray(array).save(destination/f'{name}.png')
    # Known synthetic geometry allows an automatic fixture approval; real pairs
    # require the user to review the window before exporting a workspace.
    export_workspace(prepared,destination/'workspace',reviewed=True,pair_id='synthetic_drift')
    result=prepared.result
    true_inverse=np.linalg.inv(np.vstack((transform,(0,0,1))))
    points=np.array([[60,60],[300,60],[60,300],[300,300],[192,192]],np.float64)
    errors=np.linalg.norm(cv2.transform(points[None],result.transform_after_to_before[:2])[0]-cv2.transform(points[None],true_inverse[:2])[0],axis=1)
    x0,y0,x1,y1=result.core_bounds_xyxy
    # Pipeline ncc_before describes the estimate before ECC, not the raw capture.
    raw_ncc=float(np.corrcoef(image[y0:y1,x0:x1].ravel(),moving[y0:y1,x0:x1].ravel())[0,1])
    record={'synthetic_only':True,'model_inference_performed':False,
        'injected_drift':{'rotation_deg':1.2,'scale':1.015,'translation_px':[7,-5]},
        'known_transform_landmark_max_error_px':float(errors.max()),
        'ncc_before':raw_ncc,'ncc_after':result.metrics.ncc_after,
        'note':'Synthetic integration check; not accuracy on experimental SEM images.'}
    (destination/'demo_report.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return destination,prepared


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('demo_registration_workspace'))
    args=parser.parse_args();output,_=make_capture_demo(args.output)
    print(f'Demo created: {output}')
    print('Open sem_workbench.py and select the generated inputs, or run:')
    print(f'python fusion_editor.py --workspace "{output / "workspace"}"')
