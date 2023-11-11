import json
import traceback
import urllib2

import time

import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import requests
from StringIO import StringIO

#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import pylab as plt

def Hbeta(D = np.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta);
	sumP = sum(P);
	H = np.log(sumP) + beta * np.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print "Computing pairwise distances..."
	(n, d) = X.shape;
	sum_X = np.sum(np.square(X), 1);
	D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
	P = np.zeros((n, n));
	beta = np.ones((n, 1));
	logU = np.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print "Computing P-values for point ", i, " of ", n, "..."

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf;
		betamax =  np.inf;
		Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	print "Mean value of sigma: ", np.mean(np.sqrt(1 / beta));
	return P;


def pca(X = np.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print "Preprocessing the data using PCA..."
	(n, d) = X.shape;
	X = X - np.tile(np.mean(X, 0), (n, 1));
	(l, M) = np.linalg.eig(np.dot(X.T, X));
	Y = np.dot(X, M[:,0:no_dims]);
	return Y;


def tsne(X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	# Check inputs
	if isinstance(no_dims, float):
		print "Error: array X should have type float.";
		return -1;
	if round(no_dims) != no_dims:
		print "Error: number of dimensions should be an integer.";
		return -1;

	# Initialize variables
        X = pca(X, initial_dims).real;
	(n, d) = X.shape;
	max_iter = 1000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = np.random.randn(n, no_dims);
	dY = np.zeros((n, no_dims));
	iY = np.zeros((n, no_dims));
	gains = np.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	P = P + np.transpose(P);
	P = P / np.sum(P);
	P = P * 4;									# early exaggeration
	P = np.maximum(P, 1e-12);

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = np.sum(np.square(Y), 1);
		num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / np.sum(num);
		Q = np.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - np.tile(np.mean(Y, 0), (n, 1));

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = np.sum(P * np.log(P / Q));
			print "Iteration ", (iter + 1), ": error is ", C

		# Stop lying about P-values
		if iter == 100:
			P = P / 4;

	# Return solution
	return Y;

def request_(req_url, sleep_time=1):
    succeeded = False
    while not succeeded:
        try:
            print("Requesting: %s" % req_url)
            response = urllib2.urlopen(req_url)
            time.sleep(sleep_time)  # obey api rate limits
            succeeded = True
        except:
            sleep_time += 1
            traceback.print_exc()
            continue
    return response


def split_image(hero_id):
    # Dont worry, this section of script was computationally generated. I am not insane
    if hero_id == 2:
      return [0, 0]
    if hero_id == 1:
      return [32, 0]
    if hero_id == 5:
      return [64, 0]
    if hero_id == 68:
      return [96, 0]
    if hero_id == 65:
      return [128, 0]
    if hero_id == 38:
      return [160, 0]
    if hero_id == 4:
      return [192, 0]
    if hero_id == 62:
      return [224, 0]
    if hero_id == 61:
      return [256, 0]
    if hero_id == 66:
      return [288, 0]
    if hero_id == 55:
      return [320, 0]
    if hero_id == 50:
      return [352, 0]
    if hero_id == 43:
      return [384, 0]
    if hero_id == 69:
      return [416, 0]
    if hero_id == 49:
      return [448, 0]
    if hero_id == 73:
      return [480, 0]
    if hero_id == 6:
      return [0, 32]
    if hero_id == 7:
      return [32, 32]
    if hero_id == 58:
      return [64, 32]
    if hero_id == 33:
      return [96, 32]
    if hero_id == 41:
      return [128, 32]
    if hero_id == 53:
      return [160, 32]
    if hero_id == 59:
      return [192, 32]
    if hero_id == 8:
      return [224, 32]
    if hero_id == 23:
      return [256, 32]
    if hero_id == 52:
      return [288, 32]
    if hero_id == 31:
      return [320, 32]
    if hero_id == 54:
      return [352, 32]
    if hero_id == 25:
      return [384, 32]
    if hero_id == 26:
      return [416, 32]
    if hero_id == 9:
      return [448, 32]
    if hero_id == 10:
      return [480, 32]
    if hero_id == 36:
      return [0, 64]
    if hero_id == 11:
      return [32, 64]
    if hero_id == 60:
      return [64, 64]
    if hero_id == 57:
      return [96, 64]
    if hero_id == 13:
      return [128, 64]
    if hero_id == 14:
      return [160, 64]
    if hero_id == 45:
      return [192, 64]
    if hero_id == 39:
      return [224, 64]
    if hero_id == 51:
      return [256, 64]
    if hero_id == 15:
      return [288, 64]
    if hero_id == 32:
      return [320, 64]
    if hero_id == 16:
      return [352, 64]
    if hero_id == 27:
      return [384, 64]
    if hero_id == 75:
      return [416, 64]
    if hero_id == 42:
      return [448, 64]
    if hero_id == 28:
      return [480, 64]
    if hero_id == 35:
      return [0, 96]
    if hero_id == 67:
      return [32, 96]
    if hero_id == 71:
      return [64, 96]
    if hero_id == 17:
      return [96, 96]
    if hero_id == 18:
      return [128, 96]
    if hero_id == 29:
      return [160, 96]
    if hero_id == 34:
      return [192, 96]
    if hero_id == 19:
      return [224, 96]
    if hero_id == 64:
      return [256, 96]
    if hero_id == 70:
      return [288, 96]
    if hero_id == 20:
      return [320, 96]
    if hero_id == 40:
      return [352, 96]
    if hero_id == 47:
      return [384, 96]
    if hero_id == 37:
      return [416, 96]
    if hero_id == 63:
      return [448, 96]
    if hero_id == 21:
      return [480, 96]
    if hero_id == 30:
      return [0, 128]
    if hero_id == 22:
      return [32, 128]
    if hero_id == 74:
      return [64, 128]
    if hero_id == 56:
      return [96, 128]
    if hero_id == 76:
      return [128, 128]
    if hero_id == 3:
      return [160, 128]
    if hero_id == 79:
      return [192, 128]
    if hero_id == 77:
      return [224, 128]
    if hero_id == 80:
      return [256, 128]
    if hero_id == 78:
      return [288, 128]
    if hero_id == 12:
      return [416, 128]
    if hero_id == 81:
      return [448, 128]
    if hero_id == 44:
      return [480, 128]
    if hero_id == 83:
      return [0, 160]
    if hero_id == 48:
      return [32, 160]
    if hero_id == 84:
      return [64, 160]
    if hero_id == 72:
      return [96, 160]
    if hero_id == 86:
      return [128, 160]
    if hero_id == 91:
      return [160, 160]
    if hero_id == 87:
      return [192, 160]
    if hero_id == 85:
      return [224, 160]
    if hero_id == 89:
      return [256, 160]
    if hero_id == 46:
      return [288, 160]
    if hero_id == 88:
      return [320, 160]
    if hero_id == 90:
      return [352, 160]
    if hero_id == 92:
      return [384, 160]
    if hero_id == 97:
      return [416, 160]
    if hero_id == 82:
      return [448, 160]
    if hero_id == 96:
      return [480, 160]
    if hero_id == 93:
      return [0, 192]
    if hero_id == 94:
      return [32, 192]
    if hero_id == 98:
      return [64, 192]
    if hero_id == 95:
      return [96, 192]
    if hero_id == 100:
      return [128, 192]
    if hero_id == 99:
      return [160, 192]
    if hero_id == 101:
      return [192, 192]
    if hero_id == 103:
      return [224, 192]
    if hero_id == 102:
      return [256, 192]
    if hero_id == 106:
      return [288, 192]
    if hero_id == 104:
      return [320, 192]
    if hero_id == 107:
      return [352, 192]
    if hero_id == 109:
      return [384, 192]
    if hero_id == 110:
      return [416, 192]
    if hero_id == 105:
      return [448, 192]
    if hero_id == 111:
      return [64, 224]
    if hero_id == 112:
      return [480, 192]
    if hero_id == 113:
      return [96, 224]
    if hero_id == 108:
      return [0, 224]
    if hero_id == 114:
      return [32, 224]


def main():
    inputs = []
    outputs = []
    info = json.load(request_("https://api.stratz.com/api/v1/hero/directory/detail"))
    keys = ["courierPurchase", "courierUpgrade", "courierKills", "apm", "casts", "abilityCasts", "kills", "deaths",
            "assists", "goldEarned", "networth", "xp", "cs", "dn", "neutrals", "is10kGoldComeback", "heroDamage",
            "towerDamage", "physicalDamage", "magicalDamage", "disableCount", "disableDuration", "stunCount",
            "stunDuration", "slowCount", "slowDuration", "healingSelf", "healingAllies", "invisibleCount", "runePower",
            "runeBounty", "wardObs", "wardSentry", "wardObsKilled", "level", "maxHp", "maxMp", "str", "agi", "int",
            "hpRegen", "mpRegen",
            "campsStacked", "supportGold", "ancients", "goldLost", "goldFed", "weakenCount",
            "weakenDuration", "physicalItemDamage", "magicalItemDamage", "healingItemSelf", "healingItemAllies",
            "attackDamage", "castDamage", "damage", "kdaAverage"]

    averages = {}
    sds = {}
    for i, k in enumerate(keys):
        tmp = []
        for hero in info["heroes"]:
            tmp.append(hero["heroTimeDetail"]["events"][0][k])
        averages[k] = sum(tmp) / len(tmp)
        sds[k] = np.std(tmp)

    for hero in info["heroes"]:
        outputs.append(hero["heroId"])
        d = hero["heroTimeDetail"]["events"][0]
        new_input = []

        for k in keys:
            new_input.append((d[k] - averages[k]) / sds[k])
        item_data = json.load(request_("https://api.stratz.com/api/v1/hero/%s/purchasePattern" % hero["heroId"]))
        items = np.zeros(265)

        period = item_data["earlyGame"]
        # Slice at 4 to only take into account top 4 items
        for i, event in enumerate(sorted(period["events"], key=lambda x: x["matchCount"], reverse=True)[:4]):
            items[event["itemId"] - 1] += (4 - i) * 0.1

        period = item_data["midGame"]
        for i, event in enumerate(sorted(period["events"], key=lambda x: x["matchCount"], reverse=True)[:4]):
            items[event["itemId"] - 1] += (4 - i) * 0.1

        period = item_data["lateGame"]
        for i, event in enumerate(sorted(period["events"], key=lambda x: x["matchCount"], reverse=True)[:4]):
            items[event["itemId"] - 1] += (4 - i) * 0.1

        mean = np.mean(items)
        std = np.std(items)
        # because of the 'spareseness'? of item entries I add the extra 10 to avoid pca kind of exploding, when it gets large entries.
        # remove the 10 and youll see what I mean
        # didnt normalise against other heroes like I did with other stats like cs,
        # because the 'points' method for each item means each hero already has same basetotal item values
        for i in items:
            new_input.append(((i - mean) / (std * 10)))

        input_size = len(new_input)
        inputs.append(new_input)

    print "Running t-SNE on Stratz Dota 2 data"

    X = np.array(inputs)
    # labels = np.array(outputs)  # not actually used if we just save heroes in id-order

    def plot(Y):
        fig, ax = plt.subplots()
        ax.scatter(Y[:, 0], Y[:, 1], 20)
        # warning the suffix on the end might change. however downloading a whole sheet is so much faster than individual images
        # thought better to leave like this
        url = "https://stratz.com/assets/img/icons/map/minimap_hero_sheet.955ff8aa.png"
        response = requests.get(url)
        img = Image.open(StringIO(response.content))
        for i in range(113):
            j = i + 1
            if j >= 24:  # because valve decided to skip id 24....yeah...
                j += 1
            x, y = split_image(j)
            imagebox = OffsetImage(img.crop((x, y, x+32, y+32)), zoom=0.8)
            ab = AnnotationBbox(imagebox, (Y[i, 0], Y[i, 1]), frameon=False,
                                xycoords='data')
            ax.add_artist(ab)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    Y = tsne(X, 2, input_size, 15)

    plot(Y)


if __name__ == "__main__":
    main()