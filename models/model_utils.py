

from models.meow_experiment.kitten_meow_1 import M1, M2, M3, M4
from models.csrnet import CSRNet
from models.meow_experiment.ccnn_tail import BigTailM1, BigTailM2, BigTail3, BigTail4, BigTail5, BigTail6, BigTail7, BigTail8, BigTail6i, BigTail9i
from models.meow_experiment.ccnn_tail import BigTail11i, BigTail10i, BigTail12i, BigTail13i, BigTail14i, BigTail15i
from models.meow_experiment.ccnn_head import H1, H2, H3, H3i, H4i
from models.meow_experiment.kitten_meow_1 import H1_Bigtail3
from models import CustomCNNv2, CompactCNNV7
from models.compact_cnn import CompactCNNV8, CompactCNNV9, CompactCNNV7i

def create_model(model_name):
    model = None
    if model_name == "M1":
        model = M1()
    elif model_name == "M2":
        model = M2()
    elif model_name == "M3":
        model = M3()
    elif model_name == "M4":
        model = M4()
    elif model_name == "CustomCNNv2":
        model = CustomCNNv2()
    elif model_name == "BigTailM1":
        model = BigTailM1()
    elif model_name == "BigTailM2":
        model = BigTailM2()
    elif model_name == "BigTail3":
        model = BigTail3()
    elif model_name == "BigTail4":
        model = BigTail4()
    elif model_name == "BigTail5":
        model = BigTail5()
    elif model_name == "BigTail6":
        model = BigTail6()
    elif model_name == "BigTail6i":
        model = BigTail6i()
    elif model_name == "BigTail9i":
        model = BigTail9i()
    elif model_name == "BigTail10i":
        model = BigTail10i()
    elif model_name == "BigTail11i":
        model = BigTail11i()
    elif model_name == "BigTail12i":
        model = BigTail12i()
    elif model_name == "BigTail13i":
        model = BigTail13i()
    elif model_name == "BigTail14i":
        model = BigTail14i()
    elif model_name == "BigTail15i":
        model = BigTail15i()
    elif model_name == "BigTail7":
        model = BigTail7()
    elif model_name == "BigTail8":
        model = BigTail8()
    elif model_name == "H1":
        model = H1()
    elif model_name == "H2":
        model = H2()
    elif model_name == "H3":
        model = H3()
    elif model_name == "H3i":
        model = H3i()
    elif model_name == "H4i":
        model = H4i()
    elif model_name == "H1_Bigtail3":
        model = H1_Bigtail3()
    elif model_name == "CompactCNNV7":
        model = CompactCNNV7()
    elif model_name == "CompactCNNV7i":
        model = CompactCNNV7i()
    elif model_name == "CompactCNNV8":
        model = CompactCNNV8()
    elif model_name == "CompactCNNV9":
        model = CompactCNNV9()
    elif model_name == "CSRNet":
        model = CSRNet()
    else:
        print("error: you didn't pick a model")
    return model