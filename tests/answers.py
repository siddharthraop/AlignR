"""Answers a good model would give for the TPU7x set, used by the scripted model in tests."""
import copy

INTRO = {
    "title": "{product-tpu7x}",
    "shortdesc": "{product-tpu7x} is for the largest training and inference workloads. Each chip has {tpu7x-tensorcores} "
                 "TensorCores and {tpu7x-sparsecores} SparseCores, and a pod holds up to {tpu7x-chips-per-pod} chips.",
    "body": {"body": {"title": "", "notes": ["note-tpu-api-tpu7x", "note-tensorflow-tpu7x"],
                      "paras": [{"text": "Use the specification topic to size a slice, the configurations topic to choose a "
                                         "slice shape, and the VM types topic to see what each VM provides.", "sources": ["S5"]}]}},
    "questions": [],
}

SPEC_ROWS = [("Chips per pod", "tpu7x-chips-per-pod", "none"), ("Peak compute per chip, BF16", "tpu7x-peak-bf16", "TFLOPs"),
             ("Peak compute per chip, FP8", "tpu7x-peak-fp8", "TFLOPs"), ("HBM capacity per chip", "tpu7x-hbm-capacity", "GiB"),
             ("HBM bandwidth per chip", "tpu7x-hbm-bandwidth", "GBps"), ("TensorCores per chip", "tpu7x-tensorcores", "none"),
             ("SparseCores per chip", "tpu7x-sparsecores", "none"),
             ("Bidirectional ICI bandwidth per chip", "tpu7x-ici-bandwidth", "GBps"),
             ("DCN bandwidth per chip", "tpu7x-dcn-per-chip", "Gbps")]

SPEC = {
    "title": "{product-tpu7x} specifications",
    "shortdesc": "Per-chip and per-pod specifications for {product-tpu7x}.",
    "body": {"system-architecture": {
        "title": "System architecture", "notes": [],
        "table": {"rows": [{"label": l, "key": k, "unit": u} for l, k, u in SPEC_ROWS]},
        "paras": [{"text": "For the memory layout of each chip, see the memory hierarchy topic.", "sources": ["S1"]}]}},
    "questions": [{"for": "writer", "about": "C1", "text": "v6e gives HBM capacity in GB and v5p in GiB. Which unit is right for v6e?"},
                  {"for": "engineer", "about": "C2", "text": "The table gives 7380 GBps and the memory section about 7.37 TB/s. Which is correct?"}],
}

CONFIG = {
    "title": "{product-tpu7x} supported configurations",
    "shortdesc": "The slice shapes you can request for {product-tpu7x}, and how many chips and hosts each uses.",
    "body": {"slice-shapes": {
        "title": "Slice shapes", "notes": ["note-slice-shape"],
        "table": {"columns": ["Shape", "Chips", "Hosts", "VMs", "Cubes", "Scope"]},
        "paras": [{"text": "Shapes also follow the general form `AxBxC`, where each dimension is a multiple of 4, up to "
                           "{tpu7x-chips-per-pod} chips. Every shape uses the {tpu7x-machine-type} machine type.", "sources": ["S3"]},
                  {"text": "Slices larger than {tpu7x-max-api-slice} chips require contacting your account team.", "sources": ["S3"]}]}},
    "questions": [],
}

VM = {
    "title": "{product-tpu7x} VM types",
    "shortdesc": "The VM types available for {product-tpu7x} and the host resources each one provides.",
    "body": {"vm-types": {"title": "VM types", "notes": [],
                          "table": {"columns": ["Machine type", "Chips per VM", "vCPUs", "RAM (GB)"]},
                          "paras": [{"text": "Each set of four chips connects to one CPU host, and every slice uses full-host, "
                                             "4-chip VMs.", "sources": ["S4"]}]}},
    "questions": [],
}

MEMORY = {
    "title": "{product-tpu7x} memory hierarchy",
    "shortdesc": "How high-bandwidth memory (HBM) and host memory are organized on a {product-tpu7x} chip.",
    "outline": [{"title": "High-bandwidth memory", "facts": ["F8"]}, {"title": "Dual-chiplet design", "facts": ["F8"]},
                {"title": "Host memory", "facts": ["F9"]}],
    "questions": [{"for": "engineer", "about": "C1", "text": "Is the HBM bandwidth 7380 GBps or approximately 7.37 TB/s?"}],
}

GOOD = {"tpu7x-intro": INTRO, "tpu7x-specifications": SPEC, "tpu7x-configurations": CONFIG, "tpu7x-vm-types": VM,
        "tpu7x-memory": MEMORY}


def good():
    return {k: [copy.deepcopy(v)] for k, v in GOOD.items()}
