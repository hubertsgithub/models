import pylab
from utils.get_dataset_colormap import create_minc_segmentation_label_colormap

MINC_CATEGORIES =\
["brick",
"carpet",
"ceramic",
"fabric",
"foliage",
"food",
"glass",
"hair",
"leather",
"metal",
"mirror",
"other",
"painted",
"paper",
"plastic",
"polishedstone",
"skin",
"sky",
"stone",
"tile",
"wallpaper",
"water",
"wood"]

c = create_minc_segmentation_label_colormap()
c = [[float(c_ij)/255 for c_ij in c_i] for c_i in c]

fig = pylab.figure()
figlegend = pylab.figure(figsize=(3,5))
ax = fig.add_subplot(111)

lines = []
for i in range(len(MINC_CATEGORIES)):
    lines.append(ax.scatter(0, 0, lw=0.5, s=300, marker='o', edgecolor='k', color=c[i]))
    #lines.append(ax.scatter(0, 0, lw=0.5, s=300, marker='o', edgecolor='k'))

figlegend.legend(lines, MINC_CATEGORIES, 'center', scatterpoints=1)
figlegend.savefig('./datasets/legend.png')



