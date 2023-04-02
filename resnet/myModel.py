import torch.nn
import torchvision
from torchvision import transforms
import torch.nn as nn

classes_names = ["黄足黄守瓜Aulacophora indica (Gmelin)", "烟粉虱Bemisia tabaci (Gennadius)",
                 "稻赤斑沫蝉Callitettix versicolor (Fabricius)", "角蜡蚧Ceroplastes ceriferus (Anderson)",
                 "油菜茎象甲Ceutorhynchus asper Roelofs", "豆突眼长蝽Chauliops fallax Scott",
                 "二化螟Chilo supperssalis (Walker)", "豌豆彩潜蝇Chromatomyia horticola(Goureau)",
                 "大青叶蝉Cicadella viridis (Linnaeus)", "稻棘缘蝽Cletus punctiger (Dallas)",
                 "悬铃木方翅网蝽Corythucha ciliata (Say)", "菊方翅网蝽Corythucha marmorata(Uhler)",
                 "稻铁甲Dicladispa armigera (Olivier)", "红袖蜡蝉Diostrombus politus Uhler",
                 "小麦叶蜂Dolerus tritici Chu", "斑须蝽Dolycoris baccarum (Linnaeus)",
                 "栗瘿蜂Dryocosmus KuriphilusYasumatsu", "小绿叶蝉Empoasca flavescens (Fabricius)",
                 "菜蝽Eurydema dominulus (Scopoli)", "赤条蝽Graphosoma rubrolineata (Westwood)",
                 "茶翅蝽Halyomorpha halys (Stål)", "乌桕癞皮瘤蛾Iscadia inexacta (Walker, 1858)",
                 "灰飞虱Laodelphax striatellus (Fallén)", "大稻缘蝽Leptocorisa acuta (Thunberg)",
                 "葱黄寡毛跳甲Luperomorpha suturalis Chen", "斑衣蜡蝉Lycorma delicatula (White)",
                 "豆荚野螟Maruca testulalis Gryer", "稻绿蝽Nezara viridula (Linnaeus)",
                 "褐飞虱Nilaparvata lugens (Stål)", "黄曲条跳甲Phyllotreta striolata (Fabricius)",
                 "菜粉蝶Pieris rapae (Linnaeus)", "小菜蛾Plutella xylostella (Linnaeus)",
                 "台湾黄毒蛾Porthesia taiwana Shiraki", "点蜂缘蝽Riptortus pedestris (Fabricius)",
                 "稻黑蝽Scotinophara lurida (Burmeister)", "大螟Sesamia inferens (Walker)",
                 "尘污灯蛾Spilosoma obliqua (Walker)", "斜纹夜蛾Spodoptera litura (Fabricius)",
                 "广二星蝽Stollia ventralis (Westwood)", "油菜叶露尾甲Strongyllodes variegatus (Fairmaire)"
                 ]  # todo 修改类名
input_size = 256  # todo
dropout = 0.5  # todo
num_classes = 40  # todo 修改分类数
model_path = './resnet_unzip_D0.pt'
device = torch.device('cpu')
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_model():
    model = torchvision.models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, num_classes))
    #   加载模型参数
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model
