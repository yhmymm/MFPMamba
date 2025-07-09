import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class MambaIRModel(SRModel):
    """MambaIR models for image restoration."""
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = (h // 200 + 1)
        split_token_w = (w // 200 + 1)
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        # 使图像的高度和宽度都能被2整除
        # if (h + mod_pad_h) % 2 != 0:
        #     mod_pad_h += 1
        # if (w + mod_pad_w) % 2 != 0:
        #     mod_pad_w += 1
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        # print('MambaIR模型输入图片大小:', img.size())
        split_h = H // split_token_h
        split_w = W // split_token_w
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                elif i == ral - 1:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                else:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j * split_w, (j + 1) * split_w)
                elif j == 0:
                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                elif j == row - 1:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                else:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])

        # Ensure height and width of chops are divisible by 2
        for i in range(len(img_chops)):
            chop = img_chops[i]
            _, _, chop_h, chop_w = chop.size()
            if chop_h % 2 != 0 or chop_w % 2 != 0:
                pad_h = 1 if chop_h % 2 != 0 else 0
                pad_w = 1 if chop_w % 2 != 0 else 0
                img_chops[i] = F.pad(chop, (0, pad_w, 0, pad_h), 'reflect')

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    # print('chop=======', chop.size())
                    out = self.net_g_ema(chop)
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    # print('chop=======', chop.size())
                    # out, g_list, l_list = self.net_g(chop)
                    # g_list.clear()
                    # l_list.clear()
                    out = self.net_g(chop)
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
