import clip
import copy
import math
import torch
import tempfile
import pydiffvg
import numpy as np
# import torch.nn.functional as F  # to pay respects
from pathlib import Path

from src.geometry import initialize_curves
from src.utils import checkin, save_img, pil_resize_long_edge_to, pil_to_np, np_to_tensor, rgb_to_yuv, \
    get_image_augmentation


def sample_indices(feat_content, feat_style):
    const = 128 ** 2  # 32k or so
    big_size = feat_content.shape[2] * feat_content.shape[3]  # num feaxels

    stride_x = int(max(math.floor(math.sqrt(big_size // const)), 1))
    offset_x = np.random.randint(stride_x)
    stride_y = int(max(math.ceil(math.sqrt(big_size // const)), 1))
    offset_y = np.random.randint(stride_y)
    xx, xy = np.meshgrid(np.arange(feat_content.shape[2])[offset_x::stride_x],
                         np.arange(feat_content.shape[3])[offset_y::stride_y])

    xx = xx.flatten()
    xy = xy.flatten()
    return xx, xy


def spatial_feature_extract(feat_result, feat_content, xx, xy):
    l2, l3 = [], []
    device = feat_result[0].device

    # for each extracted layer
    for i in range(len(feat_result)):
        fr = feat_result[i]
        fc = feat_content[i]

        # hack to detect reduced scale
        if i > 0 and feat_result[i - 1].size(2) > feat_result[i].size(2):
            xx = xx / 2.0
            xy = xy / 2.0

        # go back to ints and get residual
        xxm = np.floor(xx).astype(np.float32)
        xxr = xx - xxm

        xym = np.floor(xy).astype(np.float32)
        xyr = xy - xym

        # do bilinear resample
        w00 = torch.from_numpy((1. - xxr) * (1. - xyr)).float().view(1, 1, -1, 1).to(device)
        w01 = torch.from_numpy((1. - xxr) * xyr).float().view(1, 1, -1, 1).to(device)
        w10 = torch.from_numpy(xxr * (1. - xyr)).float().view(1, 1, -1, 1).to(device)
        w11 = torch.from_numpy(xxr * xyr).float().view(1, 1, -1, 1).to(device)

        xxm = np.clip(xxm.astype(np.int32), 0, fr.size(2) - 1)
        xym = np.clip(xym.astype(np.int32), 0, fr.size(3) - 1)

        s00 = xxm * fr.size(3) + xym
        s01 = xxm * fr.size(3) + np.clip(xym + 1, 0, fr.size(3) - 1)
        s10 = np.clip(xxm + 1, 0, fr.size(2) - 1) * fr.size(3) + (xym)
        s11 = np.clip(xxm + 1, 0, fr.size(2) - 1) * fr.size(3) + np.clip(xym + 1, 0, fr.size(3) - 1)

        fr = fr.view(1, fr.size(1), fr.size(2) * fr.size(3), 1)
        fr = fr[:, :, s00, :].mul_(w00).add_(fr[:, :, s01, :].mul_(w01)).add_(fr[:, :, s10, :].mul_(w10)).add_(
            fr[:, :, s11, :].mul_(w11))

        fc = fc.view(1, fc.size(1), fc.size(2) * fc.size(3), 1)
        fc = fc[:, :, s00, :].mul_(w00).add_(fc[:, :, s01, :].mul_(w01)).add_(fc[:, :, s10, :].mul_(w10)).add_(
            fc[:, :, s11, :].mul_(w11))

        l2.append(fr)
        l3.append(fc)

    x_st = torch.cat([li.contiguous() for li in l2], 1)
    c_st = torch.cat([li.contiguous() for li in l3], 1)

    xx = torch.from_numpy(xx).view(1, 1, x_st.size(2), 1).float().to(device)
    yy = torch.from_numpy(xy).view(1, 1, x_st.size(2), 1).float().to(device)

    x_st = torch.cat([x_st, xx, yy], 1)
    c_st = torch.cat([c_st, xx, yy], 1)
    return x_st, c_st


def pairwise_distances_cos(x, y):
    x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))
    dist = 1. - torch.mm(x, y_t) / x_norm / y_norm
    return dist


def pairwise_distances_sq_l2(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 1e-5, 1e5) / x.size(1)


def distmat(x, y, cos_d=True):
    if cos_d:
        M = pairwise_distances_cos(x, y)
    else:
        M = torch.sqrt(pairwise_distances_sq_l2(x, y))
    return M


def content_loss(feat_result, feat_content):
    d = feat_result.size(1)

    X = feat_result.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    Y = feat_content.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    Y = Y[:, :-2]
    X = X[:, :-2]
    # X = X.t()
    # Y = Y.t()

    Mx = distmat(X, X)
    Mx = Mx  # /Mx.sum(0, keepdim=True)

    My = distmat(Y, Y)
    My = My  # /My.sum(0, keepdim=True)

    d = torch.abs(Mx - My).mean()  # * X.shape[0]
    return d


def style_loss(X, Y, cos_d=True):
    d = X.shape[1]

    if d == 3:
        X = rgb_to_yuv(X.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        Y = rgb_to_yuv(Y.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
    else:
        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    # Relaxed EMD
    CX_M = distmat(X, Y, cos_d=True)

    if d == 3: CX_M = CX_M + distmat(X, Y, cos_d=False)

    m1, m1_inds = CX_M.min(1)
    m2, m2_inds = CX_M.min(0)

    remd = torch.max(m1.mean(), m2.mean())

    return remd


def moment_loss(X, Y, moments=[1, 2]):
    loss = 0.
    X = X.squeeze().t()
    Y = Y.squeeze().t()

    mu_x = torch.mean(X, 0, keepdim=True)
    mu_y = torch.mean(Y, 0, keepdim=True)
    mu_d = torch.abs(mu_x - mu_y).mean()

    if 1 in moments:
        # print(mu_x.shape)
        loss = loss + mu_d

    if 2 in moments:
        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

        # print(X_cov.shape)
        # exit(1)

        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = loss + D_cov

    return loss


def calculate_loss(feat_result, feat_content, feat_style, indices, content_weight, moment_weight=1.0):
    # spatial feature extract
    num_locations = 1024
    spatial_result, spatial_content = spatial_feature_extract(feat_result, feat_content, indices[0][:num_locations],
                                                              indices[1][:num_locations])
    # loss_content = content_loss(spatial_result, spatial_content)

    d = feat_style.shape[1]
    spatial_style = feat_style.view(1, d, -1, 1)
    feat_max = 3 + 2 * 64 + 128 * 2 + 256 * 3 + 512 * 2  # (sum of all extracted channels)

    loss_remd = style_loss(spatial_result[:, :feat_max, :, :], spatial_style[:, :feat_max, :, :])

    loss_moment = moment_loss(spatial_result[:, :-2, :, :], spatial_style, moments=[1, 2])  # -2 is so that it can fit?
    # palette matching
    content_weight_frac = 1. / max(content_weight, 1.)
    loss_moment += content_weight_frac * style_loss(spatial_result[:, :3, :, :], spatial_style[:, :3, :, :])

    loss_style = loss_remd + moment_weight * loss_moment
    # print(f'Style: {loss_style.item():.3f}, Content: {loss_content.item():.3f}')

    style_weight = 1.0 + moment_weight
    loss_total = (loss_style) / (content_weight + style_weight)
    return loss_total


def render_drawing(shapes, shape_groups, canvas_width, canvas_height, n_iter, save=False):
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, n_iter, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()) * (
                1 - img[:, :, 3:4])
    if save:
        pydiffvg.imwrite(img.cpu(), '/content/res/iter_{}.png'.format(int(n_iter)), gamma=1.0)
    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
    return img


def style_clip_draw(prompt, style_path, model, extractor,
                    num_paths=256, num_iter=1000, max_width=50,
                    num_augs=4, style_weight=1.,
                    neg_prompt=None, neg_prompt_2=None,
                    use_normalized_clip=True,
                    debug=False, st_pbar=None, black_and_white=True, uniform_width=True):
    '''
    Perform StyleCLIPDraw using a given text prompt and style image
    args:
        prompt (str) : Text prompt to draw
        style_path(str) : Style image path or url
    kwargs:
        num_paths (int) : Number of brush strokes
        num_iter(int) : Number of optimization iterations
        max_width(float) : Maximum width of a brush stroke in pixels
        num_augs(int) : Number of image augmentations
        style_weight=(float) : What to multiply the style loss by
        neg_prompt(str) : Negative prompt. None if you don't want it
        neg_prompt_2(str) : Negative prompt. None if you don't want it
        use_normalized_clip(bool)
        debug(bool) : Print intermediate canvases and losses for debugging
    return
        np.ndarray(canvas_height, canvas_width, 3)
    '''
    out_path = Path(tempfile.mkdtemp()) / "out.png"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    text_input = clip.tokenize(prompt).to(device)

    if neg_prompt is not None: text_input_neg1 = clip.tokenize(neg_prompt).to(device)
    if neg_prompt_2 is not None: text_input_neg2 = clip.tokenize(neg_prompt_2).to(device)

    # Calculate features
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        if neg_prompt is not None: text_features_neg1 = model.encode_text(text_input_neg1)
        if neg_prompt_2 is not None: text_features_neg2 = model.encode_text(text_input_neg2)

    canvas_width, canvas_height = 224, 224

    # Image Augmentation Transformation
    augment_trans = get_image_augmentation(use_normalized_clip)

    # Initialize Random Curves
    shapes, shape_groups = initialize_curves(num_paths, canvas_width, canvas_height, black_and_white)

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        # print(group.stroke_color)
        color_vars.append(group.stroke_color)

    # Optimizers
    lr = 1
    points_optim = torch.optim.Adam(points_vars, lr=1.0 * lr)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1 * lr)
    color_optim = torch.optim.Adam(color_vars, lr=0.01 * lr)

    style_pil = style_path.convert("RGB")
    style_pil = pil_resize_long_edge_to(style_pil, canvas_width)
    style_np = pil_to_np(style_pil)
    style = (np_to_tensor(style_np, "normal").to(device) + 1) / 2

    # Extract style features from style image
    feat_style = None
    for i in range(5):
        with torch.no_grad():
            # r is region of interest (mask)
            feat_e = extractor.forward_samples_hypercolumn(style, samps=1000)
            feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)

    # Run the main optimization loop
    for t in range(num_iter):

        # Anneal learning rate (makes videos look cleaner)
        if t == int(num_iter * 0.5):
            for g in points_optim.param_groups:
                g['lr'] = 0.4
        if t == int(num_iter * 0.75):
            for g in points_optim.param_groups:
                g['lr'] = 0.1

        points_optim.zero_grad()
        if not uniform_width:
            width_optim.zero_grad()
        if not black_and_white:
            color_optim.zero_grad()

        img = render_drawing(shapes, shape_groups, canvas_width, canvas_height, t, save=(t % 5 == 0))

        loss = 0
        img_augs = []
        if t < .9 * num_iter:
            for n in range(num_augs):
                img_augs.append(augment_trans(img))
            im_batch = torch.cat(img_augs)
            image_features = model.encode_image(im_batch)
            for n in range(num_augs):
                loss -= torch.cosine_similarity(text_features, image_features[n:n + 1], dim=1)
                if neg_prompt is not None: loss += torch.cosine_similarity(text_features_neg1, image_features[n:n + 1],
                                                                           dim=1) * 0.3
                if neg_prompt_2 is not None: loss += torch.cosine_similarity(text_features_neg2,
                                                                             image_features[n:n + 1], dim=1) * 0.3

        # Do style optimization
        feat_content = extractor(img)

        xx, xy = sample_indices(feat_content[0], feat_style)

        np.random.shuffle(xx)
        np.random.shuffle(xy)

        styleloss = calculate_loss(feat_content, feat_content, feat_style, [xx, xy], 0)

        loss += styleloss * style_weight

        loss.backward()
        points_optim.step()
        if not uniform_width:
            width_optim.step()
        if not black_and_white:
            color_optim.step()

        for path in shapes:
            path.stroke_width.data.clamp_(1.0, max_width)
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

        if t % 20 == 0:
            with torch.no_grad():
                shapes_resized = copy.deepcopy(shapes)
                for i in range(len(shapes)):
                    shapes_resized[i].stroke_width = shapes[i].stroke_width * 4
                    for j in range(len(shapes[i].points)):
                        shapes_resized[i].points[j] = shapes[i].points[j] * 4
                img = render_drawing(shapes_resized, shape_groups, canvas_width * 4, canvas_height * 4, t)

                data_obj = {
                    'shapes': shapes_resized,
                    'shape_groups': shape_groups,
                    'width': canvas_width * 4,
                    'height': canvas_height * 4,
                }

                yield checkin(img.detach().cpu().numpy()[0], out_path), data_obj
                print('Iteration:', t, '\tRender loss:', loss.item())

        if st_pbar is not None:
            st_pbar.progress(t / num_iter)

    with torch.no_grad():
        shapes_resized = copy.deepcopy(shapes)
        for i in range(len(shapes)):
            shapes_resized[i].stroke_width = shapes[i].stroke_width * 4
            for j in range(len(shapes[i].points)):
                shapes_resized[i].points[j] = shapes[i].points[j] * 4
        img = render_drawing(shapes_resized, shape_groups, canvas_width * 4, canvas_height * 4, t).detach().cpu().numpy()[0]
        save_img(img, str(out_path))
        data_obj = {
            'shapes': shapes_resized,
            'shape_groups': shape_groups,
            'width': canvas_width * 4,
            'height': canvas_height * 4,
        }
        yield out_path, data_obj


def style_clip_draw_slow(prompt, style_path, model, extractor,
                         num_paths=256, num_iter=1000, max_width=50,
                         num_augs=4, style_opt_freq=5, style_opt_iter=50, style_weight=1.,
                         neg_prompt=None, neg_prompt_2=None,
                         use_normalized_clip=True,
                         debug=False, st_pbar=None, ):
    '''
    Perform StyleCLIPDraw using a given text prompt and style image
    args:
        prompt (str) : Text prompt to draw
        style_path(str) : Style image path or url
    kwargs:
        num_paths (int) : Number of brush strokes
        num_iter(int) : Number of optimization iterations
        max_width(float) : Maximum width of a brush stroke in pixels
        num_augs(int) : Number of image augmentations
        style_opt_freq(int) : How often to do style optimization. Low value is high frequency
        style_opt_iter(int) : How many iterations to do in the style optimization loop
        neg_prompt(str) : Negative prompt. None if you don't want it
        neg_prompt_2(str) : Negative prompt. None if you don't want it
        use_normalized_clip(bool)
        debug(bool) : Print intermediate canvases and losses for debugging
    return
        np.ndarray(canvas_height, canvas_width, 3)
    '''
    out_path = Path(tempfile.mkdtemp()) / "out.png"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    text_input = clip.tokenize(prompt).to(device)

    if neg_prompt is not None: text_input_neg1 = clip.tokenize(neg_prompt).to(device)
    if neg_prompt_2 is not None: text_input_neg2 = clip.tokenize(neg_prompt_2).to(device)

    # Calculate features
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        if neg_prompt is not None: text_features_neg1 = model.encode_text(text_input_neg1)
        if neg_prompt_2 is not None: text_features_neg2 = model.encode_text(text_input_neg2)

    canvas_width, canvas_height = 224, 224

    # Image Augmentation Transformation
    augment_trans = get_image_augmentation(use_normalized_clip)

    # Initialize Random Curves
    shapes, shape_groups = initialize_curves(num_paths, canvas_width, canvas_height)

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    # Optimizers
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # points_vars = [l.data.requires_grad_() for l in points_vars]
    points_optim_style = torch.optim.RMSprop(points_vars, lr=0.1)
    width_optim_style = torch.optim.RMSprop(stroke_width_vars, lr=0.1)
    color_optim_style = torch.optim.RMSprop(color_vars, lr=0.01)

    style_pil = style_path.convert("RGB")
    style_pil = pil_resize_long_edge_to(style_pil, canvas_width)
    style_np = pil_to_np(style_pil)
    style = (np_to_tensor(style_np, "normal").to(device) + 1) / 2
    # style_pil = pil_loader(style_path) if os.path.exists(style_path) else pil_loader_internet(style_path)
    # style_pil = pil_resize_long_edge_to(style_pil, canvas_width)
    # style_np = pil_to_np(style_pil)
    # style = (np_to_tensor(style_np, "normal").to(device) + 1) / 2
    # extractor = Vgg16_Extractor(space="normal").to(device)

    # Extract style features from style image
    feat_style = None
    for i in range(5):
        with torch.no_grad():
            # r is region of interest (mask)
            feat_e = extractor.forward_samples_hypercolumn(style, samps=1000)
            feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)

    # Run the main optimization loop
    # for t in range(num_iter) if debug else tqdm(range(num_iter)):
    for t in range(num_iter):

        # Anneal learning rate (makes videos look cleaner)
        if t == int(num_iter * 0.5):
            for g in points_optim.param_groups:
                g['lr'] = 0.4
        if t == int(num_iter * 0.75):
            for g in points_optim.param_groups:
                g['lr'] = 0.1

        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()

        img = render_drawing(shapes, shape_groups, canvas_width, canvas_height, t, save=(t % 5 == 0))

        loss = 0
        img_augs = []
        for n in range(num_augs):
            img_augs.append(augment_trans(img))
        im_batch = torch.cat(img_augs)
        image_features = model.encode_image(im_batch)
        for n in range(num_augs):
            loss -= torch.cosine_similarity(text_features, image_features[n:n + 1], dim=1)
            if neg_prompt is not None: loss += torch.cosine_similarity(text_features_neg1, image_features[n:n + 1],
                                                                       dim=1) * 0.3
            if neg_prompt_2 is not None: loss += torch.cosine_similarity(text_features_neg2, image_features[n:n + 1],
                                                                         dim=1) * 0.3

        loss.backward()
        points_optim.step()
        width_optim.step()
        color_optim.step()

        # Do style optimization from time to time and on the last iteration
        if t % style_opt_freq == 0 or (t == num_iter - 1):
            img = render_drawing(shapes, shape_groups, canvas_width, canvas_height, t)
            feat_content = extractor(img)

            xx, xy = sample_indices(feat_content[0], feat_style)  # 0 to sample over first layer extracted
            for it in range(style_opt_iter):
                styleloss = 0
                points_optim_style.zero_grad()
                width_optim_style.zero_grad()
                color_optim_style.zero_grad()

                img = render_drawing(shapes, shape_groups, canvas_width, canvas_height, t)
                feat_content = extractor(img)

                if it % 1 == 0 and it != 0:
                    np.random.shuffle(xx)
                    np.random.shuffle(xy)

                styleloss = calculate_loss(feat_content, feat_content, feat_style, [xx, xy], 0)

                styleloss.backward()
                points_optim_style.step()
                width_optim_style.step()
                color_optim_style.step()

        for path in shapes:
            path.stroke_width.data.clamp_(1.0, max_width)
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

        if t % 20 == 0:
            with torch.no_grad():
                shapes_resized = copy.deepcopy(shapes)
                for i in range(len(shapes)):
                    shapes_resized[i].stroke_width = shapes[i].stroke_width * 4
                    for j in range(len(shapes[i].points)):
                        shapes_resized[i].points[j] = shapes[i].points[j] * 4
                img = render_drawing(shapes_resized, shape_groups, canvas_width * 4, canvas_height * 4, t)

                data_obj = {
                    'shapes': shapes_resized,
                    'shape_groups': shape_groups,
                    'width': canvas_width * 4,
                    'height': canvas_height * 4,
                }

                yield checkin(img.detach().cpu().numpy()[0], out_path), data_obj
                print('Iteration:', t, '\tRender loss:', loss.item())
            # img = render_scaled(shapes, shape_groups, canvas_width, canvas_height, t=t)
            # img = render_drawing(shapes_resized, shape_groups, canvas_width * 4, canvas_height * 4, t)
            #
            # data_obj = {
            #     'shapes': shapes_resized,
            #     'shape_groups': shape_groups,
            #     'width': canvas_width * 4,
            #     'height': canvas_height * 4,
            # }
            #
            # yield checkin(img.detach().cpu().numpy()[0], out_path), data_obj
            # print('Iteration:', t, '\tRender loss:', loss.item())
            # show_img(img.detach().cpu().numpy()[0])
            # show_img(torch.cat([img.detach(), img_aug.detach()], axis=3).cpu().numpy()[0])
            # print('render loss:', loss.item())
            # print('iteration:', t)
            # with torch.no_grad():
            #     im_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            #     noun_norm = nouns_features / nouns_features.norm(dim=-1, keepdim=True)
            #     similarity = (100.0 * im_norm @ noun_norm.T).softmax(dim=-1)
            #     values, indices = similarity[0].topk(5)
            #     print("\nTop predictions:\n")
            #     for value, index in zip(values, indices):
            #         print(f"{nouns[index]:>16s}: {100 * value.item():.2f}%")

        if st_pbar is not None:
            st_pbar.progress(t / num_iter)

    with torch.no_grad():
        shapes_resized = copy.deepcopy(shapes)
        for i in range(len(shapes)):
            shapes_resized[i].stroke_width = shapes[i].stroke_width * 4
            for j in range(len(shapes[i].points)):
                shapes_resized[i].points[j] = shapes[i].points[j] * 4
        img = \
            render_drawing(shapes_resized, shape_groups, canvas_width * 4, canvas_height * 4, t).detach().cpu().numpy()[
                0]
        save_img(img, str(out_path))
        data_obj = {
            'shapes': shapes_resized,
            'shape_groups': shape_groups,
            'width': canvas_width * 4,
            'height': canvas_height * 4,
        }
        yield out_path, data_obj
