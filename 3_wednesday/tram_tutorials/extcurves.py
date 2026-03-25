from pylab import *
import numpy as np
from numpy import *
from numpy import poly1d

def extcurve_obs(wave, R_V, NH_EBV= 5.8e21, model = 'ODonnell94'):

	"""
	 Cardelli, Clayton and Mathis (1989) extinction curves, including parameters
	 
	 O'Donnell 1994 improved interstellar extinction from CCM89 for x=1.1-3.3mum-1.
	 Two models CCM89 and ODonnell only slightly different in x=1.1-3.3mum-1
		
     INPUT:
         WAVE - wavelength vector (micron)
		 RV--ratio of visual extinction to reddening, RV=AV/E(B-V)
		 NH_EBV = NH/E_BV= gas-to-dust ratio. For Galaxy, NH_EBV = 5.8e21 Hcm^2/mag
     OUTPUT:
		 CCM Extinction curves, Alambda/AV and Alambda/NH
		 
		 NOTE: A(lambda)/A(V) = 1 + 1/RV *E(lambda-V)/E(B-V) = 1 + FM(x)/R = a + b/RV, so b = E(lambda-V)/E(B-V)
    
     REVISION HISTORY:
		   Thiem Hoang: split this function into the function to compute
		   extinction curve and one function for unredded flux, 2013 Dec
		   The current code does not work for wavelen>3.3 micron or x=1/w<0.3
		
           Written W. Landsman Hughes/STX   January, 1992
           Extrapolate curve for wavelengths between 900 & 1000 A   Dec. 1993
           Use updated coefficients for near-UV from O'Donnell   Feb 1994
           Allow 3 parameter calling sequence      April 1998
           Converted to IDLV5.0                    April 1998
			
    """
	x = 1./ np.array(wave)							# Convert to inverse microns
	npts = x.size
	a = np.zeros(npts, dtype = float)
	b = np.zeros(npts, dtype = float)
    #******************************

	#2013 Jan 8: Thiem extended for waveleng > 3.3micron or x<0.3
	good = np.where( (x < 0.3))						#far-Infrared
	if len(good[0]) > 0:
		a[good] = 0.574 * x[good] ** (1.61)
		b[good] = -0.527 * x[good] ** (1.61)

													#Infrared
	good = np.where( (x >= 0.3) & (x < 1.1) )
	if len(good[0]) > 0:
		a[good] = 0.574 * x[good] ** (1.61)
		b[good] = -0.527 * x[good] ** (1.61)
    
													#Optical/NIR
	good = np.where( (x >= 1.1) & (x < 3.3) )
	if len(good[0]) > 0:  # Use new constants from O'Donnell (1994)
		y = x[good] - 1.82
		#     c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085,    $ ;Original
		#                 0.01979, -0.77530,  0.32999 ]               ;coefficients
		#     c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434,    $ ;from CCM89
		#                -0.62251,  5.30260, -2.09002 ]

		#** NOTE **:
		#  IDL poly() wants coefficients starting with A0, then A1 then ...AN where 
		#             AN is the coefficient for X^N
		#             So the coefficients are given in that order
		
		if(model == 'CCM89'):
			c1 = np.array([ 1. , 0.17699, -0.50447, -0.02427,  0.72085,0.01979, -0.77530,  0.32999 ])
			c2 = np.array([ 0.,  1.41338,  2.28305,  1.07233, -5.38434,-0.62251,  5.30260, -2.09002])
		else:
			# default model is using O'Donnell 1994 result
			c1 = np.array([1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505])        #from O'Donnell
			c2 = np.array([0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347])
		#

		#  np's poly1d wants **exactly the opposite order **
		#       so swap 'em

		# Here c1[::-1] to reverse the order of c1 to be used for poly1d PYTHON (opposite from IDL)
		a[good] = poly1d(c1[::-1])(y)
		b[good] = poly1d(c2[::-1])(y)
    #******************************
    
	good = np.where( (x >=3.3) & (x < 8))
														# Mid-UV, including UV bump: polynomial + Drude function
	if len(good[0]) > 0:    
		
		y = x[good]
		f_a = np.zeros([len(good[0])], dtype=float)    #
		f_b = np.zeros([len(good[0])], dtype=float)    #f_b = np.zeros([ngood], dtype=float32)
		
		#	good1 = np.where(ravel((y > 5.9)))[0]
		good1 = np.where(y > 5.9)
		# Thiem: removed [0] from good1 above, it works properly now
		if len(good1[0]) > 0:
			y1 = y[good1] - 5.9
			f_a[good1] = -0.04473 * y1**2 - 0.009779 * y1**3
			f_b[good1] = 0.2130 * y1**2 + 0.1207 * y1**3
		
		a[good] = 1.752 - 0.316 * y - (0.104 / ((y - 4.67)**2 + 0.341)) + f_a
		b[good] = -3.090 + 1.825 * y + (1.206 / ((y - 4.62)**2 + 0.263)) + f_b

		# a = c0 + c1*y + c2*y^2 + c3*y^3 + Drude profile (gamma), where c0-c3 and gamma are coefficients

    #   *******************************
														# Far-UV, only polynomial functions
	good = np.where( (x >= 8) & (x <= 11) )
	if len(good[0]) > 0:    
		y = x[good] - 8.
		c1 = np.array([-1.073, -0.628, 0.137, -0.070])
		c2 = np.array([13.670, 4.257, -0.420, 0.374])
		a[good] = poly1d(c1[::-1])(y)
		b[good] = poly1d(c2[::-1])(y)

	"""
	A_V = R_V * EBV
	here EBV = E(B-V) = A(V) - A(B) = 'reddening',V=0.55,B=blue=0.44mu
	NH/E(B-V)= 5.8e21 for Galactic plane
	For depending on galactic lattitude, it changes.
	So for a known R_V, one obtain
	A_lambda/NH=A_lambda/(EBV*5.8e21)=(A_lambda/A_V)*(RV/5.8e21)--eq(21.6 - BruceDraine book)
	"""
	A_lambda_AV	= (a + b / R_V)
	A_lambda_NH	= (a + b / R_V)*(R_V/NH_EBV)

	return A_lambda_AV

#end function
