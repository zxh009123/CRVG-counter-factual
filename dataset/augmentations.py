import torch
import numpy as np
import math

def HFlip(sat, grd):
    sat = torch.flip(sat, [2])
    grd = torch.flip(grd, [2])

    return sat, grd

def Rotate(sat, grd, orientation, is_polar):
    height, width = grd.shape[1], grd.shape[2]
    if orientation == 'left':
        if is_polar:
            left_sat = sat[:, :, 0:int(math.ceil(width * 0.75))]
            right_sat = sat[:, :, int(math.ceil(width * 0.75)):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, -1, [1, 2])
        left_grd = grd[:, :, 0:int(math.ceil(width * 0.75))]
        right_grd = grd[:, :, int(math.ceil(width * 0.75)):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'right':
        if is_polar:
            left_sat = sat[:, :, 0:int(math.floor(width * 0.25))]
            right_sat = sat[:, :, int(math.floor(width * 0.25)):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
        left_grd = grd[:, :, 0:int(math.floor(width * 0.25))]
        right_grd = grd[:, :, int(math.floor(width * 0.25)):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'back':
        if is_polar:
            left_sat = sat[:, :, 0:int(width * 0.5)]
            right_sat = sat[:, :, int(width * 0.5):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
            sat_rotate = torch.rot90(sat_rotate, 1, [1,2])
        left_grd = grd[:, :, 0:int(width * 0.5)]
        right_grd = grd[:, :, int(width * 0.5):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)
    
    else:
        raise RuntimeError(f"Orientation {orientation} is not implemented")

    return sat_rotate, grd_rotate

def Reverse_Rotate_Flip(sat, ground, perturb, polar):
    # Reverse process
    reverse_perturb = [None, None]
    reverse_perturb[0] = perturb[0]

    if perturb[1] == "left":
        reverse_perturb[1] = "right"
    elif perturb[1] == "right":
        reverse_perturb[1] = "left"
    else:
        reverse_perturb[1] = perturb[1]

    # print(reverse_perturb)

    # reverse process first rotate then flip
    re_sat, re_grd = Rotate(sat, ground, reverse_perturb[1], polar)

    if reverse_perturb[0] == 1:
        re_sat, re_grd = HFlip(re_sat, re_grd)

    return re_sat, re_grd


def Free_Rotation(sat, grd, degree, axis="h"):
    """
    only for polar case
    degree will be made in [0., 360.]; clockwise by default; can be negative;
    axis="h" for horizontal and "v" for vertical direction in the polar image;
        - "h" for normal (improper) rotation & flip; rel pos preserved
        - "v" change the distribution; rel pos NOT preserved
    NOTE sat & grd of shape: (bs, c, h, w)
    """
    height, width = grd.shape[2], grd.shape[3]

    degree = np.mod(degree, 360.)
    ratio = 1.- degree / 360.
    if axis == "h":
        bound = int(width * ratio)

        left_sat  = sat[:, :, :, 0:bound]
        right_sat = sat[:, :, :, bound:]

        left_grd  = grd[:, :, :, 0:bound]
        right_grd = grd[:, :, :, bound:]

        sat_rotate = torch.cat([right_sat, left_sat], dim=3)
        grd_rotate = torch.cat([right_grd, left_grd], dim=3)

    elif axis == "v":
        bound = int(height * ratio)

        left_sat  = sat[:, :, 0:bound]
        right_sat = sat[:, :, bound:]

        left_grd  = grd[:, :, 0:bound]
        right_grd = grd[:, :, bound:]

        sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    return sat_rotate, grd_rotate


def Free_Improper_Rotation(sat, grd, degree, axis="h"):
    """
    only for polar case
    degree will be made in [0., 360.]; clockwise by default; can be negative;
    axis="h" for horizontal and "v" for vertical direction in the polar image;
        - "h" for normal (improper) rotation & flip; rel pos preserved
        - "v" change the distribution; rel pos NOT preserved
    NOTE sat & grd of shape: (bs, c, h, w)
    """
    new_sat = torch.flip(sat, [3])
    new_grd = torch.flip(grd, [3])

    sat_rotate, grd_rotate = Free_Rotation(new_sat, new_grd, degree, axis=axis)

    return sat_rotate, grd_rotate


def Free_Flip(sat, grd, degree):
    """
    only for polar case
    (virtually) flip-reference is the non-polar sat-view image
    degree specifies the flip axis
    degree will be made in [0., 360.]; clockwise by default; can be negative;
    """
    # rotate by -degree
    new_sat, new_grd = Free_Rotation(sat, grd, -degree, axis="h")

    # h-flip
    new_sat = torch.flip(new_sat, [3])
    new_grd = torch.flip(new_grd, [3])

    # rotate back by degree
    new_sat, new_grd = Free_Rotation(new_sat, new_grd, degree, axis="h")
    
    return new_sat, new_grd


if __name__ == "__main__":
    sat = torch.rand(32, 16, 16, 8)
    # sat = torch.rand(32, 8, 42, 8)
    polar = False
    grd = torch.rand(32, 8, 42, 8)


    # mu_sat = sat.clone().detach()
    mu_sat = sat.clone().detach()
    mu_grd = grd.clone().detach()
    

    perturb = [0, "left"]
    if perturb[0] == 1:
        mu_sat, mu_grd = HFlip(mu_sat, mu_grd)
    mu_sat, mu_grd = Rotate(mu_sat, mu_grd, perturb[1], polar)

    print("=====before:")
    print(grd[0, 0, 0:8, 0])
    print(mu_grd[0, 0, 0:8, 0])
    print(torch.equal(sat, mu_sat))
    print(torch.equal(grd, mu_grd))

    re_sat, re_grd = Reverse_Rotate_Flip(mu_sat, mu_grd, perturb, polar)

    # # Reverse process
    # reverse_perturb = [None, None]
    # reverse_perturb[0] = perturb[0]

    # if perturb[1] == "left":
    #     reverse_perturb[1] = "right"
    # elif perturb[1] == "right":
    #     reverse_perturb[1] = "left"
    # else:
    #     reverse_perturb[1] = perturb[1]

    # # print(reverse_perturb)

    # # reverse process first rotate then flip
    # mu_sat, mu_grd = Rotate(mu_sat, mu_grd, reverse_perturb[1], polar)

    # if reverse_perturb[0] == 1:
    #     mu_sat, mu_grd = HFlip(mu_sat, mu_grd)

    print("=====after:")
    print(grd[0, 0, 0:8, 0])
    print(re_grd[0, 0, 0:8, 0])
    print(torch.equal(sat, re_sat))
    print(torch.equal(grd, re_grd))