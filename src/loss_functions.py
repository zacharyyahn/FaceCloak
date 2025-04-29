def fawkes_loss(out_emb, args):
    return args["dist_func"](out_emb, args["tgt_emb"])

# Decrease distance between self and target, and increase distance between self and closest
def triplet_loss(out_emb, args):
    return args["dist_func"](out_emb, args["tgt_emb"]) - args["dist_func"](out_emb, args["closest_emb"])
