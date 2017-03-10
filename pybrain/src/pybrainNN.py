#######################################################################
#################   Feed forward network #############################

#   Feed Forward Network Object

from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(2,3,1)

#activate network
net.activate([2,1])

#examine structure
net['in']
net['hidden0']
net['out']

#sophisticated network
#from pybrain.structure import TanhLayer
#net = buildNetwork(2, 3, 1, hiddenclass=TanhLayer)
#net['hidden0']

#   Customise Dataset
from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(2,1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))

print len(ds)

for inpt, target in ds:
    print inpt, target

print '\nInputs'
print ds['input']

print '\nOutputs/Targets'
print ds['target']

#   ds.clear();

##################  Trainers #####################
from pybrain import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
net = buildNetwork(2,3,1, bias=True, hiddenclass = TanhLayer)
trainer = BackpropTrainer(net,ds)

print trainer.train()
#    tuple containing the errors for every training epoch.
print trainer.trainUntilConvergence()

############### Feed forward networks ######################
from pybrain.structure import FeedForwardNetwork
n = FeedForwardNetwork()

#   Constructing input, output & hidden layers & giving names to the network
from pybrain.structure import LinearLayer, SigmoidLayer
inLayer = LinearLayer(2, name='in')
hiddenLayer = SigmoidLayer(3, name='hidden')
outLayer = LinearLayer(1, name='out')

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

#   Full Connection class - add connections/synapses
from pybrain.structure import FullConnection
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)
#makes our MLP usable,
n.sortModules()
print n.activate([1,2])

print n

#Recureent Connection Class -which looks back in time one timestep.
from pybrain.structure import RecurrentNetwork
n = RecurrentNetwork()
n.addInputModule(LinearLayer(2, name='in'))
n.addModule(SigmoidLayer(3, name='hidden'))
n.addOutputModule(LinearLayer(1, name='out'))
n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))

n.sortModules()
print n.activate((2, 2))
print n.activate((2, 2))
print n.activate((2, 2))
n.reset()
print n.activate((2,2))

#######################################
#########   Classification with feed forward networks
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

# ######################  graphical output ########################
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

#To have a nice dataset for visualization,
# we produce a set of points in 2D belonging to three different classes.
means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(2, 1, nb_classes=3)
for n in xrange(400):
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input, [klass])

#Randomly split the dataset into 75% training and 25% test data sets
tstdata, trndata = alldata.splitWithProportion( 0.25 )

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "\nNumber of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]


fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=3)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

for i in range(20):
    trainer.trainEpochs(1)
    trnresult = percentError(trainer.testOnClassData(),
                             trndata['class'])
    tstresult = percentError(trainer.testOnClassData(
        dataset=tstdata), tstdata['class'])

    print "epoch: %4d" % trainer.totalepochs, \
        "  train error: %5.2f%%" % trnresult, \
        "  test error: %5.2f%%" % tstresult
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)  # the highest output activation gives the class
    out = out.reshape(X.shape)
    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    hold(True) # overplot on
    for c in [0,1,2]:
        here, _ = where(tstdata['class']==c)
        plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
    if out.max()!=out.min():  # safety check against flat field
        contourf(X, Y, out)   # plot the contour
    ion()   # interactive graphics on
    draw()  # update the plot

ioff()
show()