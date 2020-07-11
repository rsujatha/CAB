import numpy as np
import scipy.special as sp

class cosmology(object):
	def __init__(self,Omega_matter = 0.276,Omega_lambda=0.724,H_0=70.,ns=0.96,sigma_8 = 0.8,kbyh=None,Tfn=None ):
		self.Omega_matter=Omega_matter
		self.rho_c_h2_msun_mpc3 = 2776*1e8    ## critical density in (msun/h)(mpc/h)**3
		self.ns = ns
		self.Omega_lambda = Omega_lambda
		self.sigma_8 = sigma_8
		self.H_0=H_0
		self.delta_c=1.686
		self.kbyh = kbyh
		self.Tfn = Tfn
		### loading fits files #################################################################
		self.m1 = np.load("../fits/p_m1_alpha.npz")
		self.s1 = np.load("../fits/p_S1_alpha.npz")
		self.m2 = np.load("../fits/p_m2_alpha.npz")
		self.s2 = np.load("../fits/p_S2_alpha.npz")

	def Wk(self,k,R):
		"""
		Fourier Transform of a Spherical Top Hat Filter
		"""
		return 3/(k*R)**3*(np.sin(k*R)-(k*R)*np.cos(k*R))	

	def GrowthFunction(self,a):
		"""
		a is the scale factor
		"""
		a=np.array(a)+1e-15
		D=np.ones(np.size(a))
		H=self.H_0*(self.Omega_matter*(a**(-3))+self.Omega_lambda)**(1/2.)
		D=(H/self.H_0)*a**(5/2.)/np.sqrt(self.Omega_matter)*sp.hyp2f1(5/6.,3/2.,11/6.,-a**3*self.Omega_lambda/self.Omega_matter)
		return D
			
	def PS(self,z):
		"""
		Input
		k in h Mpc^1
		z redshift
		T the tranfer function
		
		Outputs 
		Pk the power spectrum
		"""
		R=8.
		integrand = 1/(2*np.pi**2)*self.kbyh**(self.ns+2.)*self.Tfn**2*cosmology.Wk(self,self.kbyh,R)**2
		igrate = np.trapz(integrand,self.kbyh)
		SigmaSquare=self.sigma_8**2
		NormConst = SigmaSquare/igrate
		return NormConst*self.kbyh**self.ns*(self.Tfn)**2*cosmology.GrowthFunction(self,1./(1.+z))**2/cosmology.GrowthFunction(self,1)**2

	def PeakHeight(self,mass,z):
		"""
		Inputs:
		mass in Msunh-1
		k in h Mpc^1
		z redshift
		T the tranfer function
		"""
		R = (3/(4*np.pi)*mass/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.) ## is in units of Mpch-1
		PS = cosmology.PS(self,z)
		sigma_square = np.zeros([len(R),1])
		for i in range(0,len(R)):
			wk = cosmology.Wk(self,self.kbyh,R[i])
			sigma_square[i] = 1/(2.*np.pi**2)*np.trapz(PS*wk**2*self.kbyh**2,self.kbyh)
		nu = self.delta_c/np.sqrt(sigma_square)
		return nu.flatten()
	
	def T10(self,argument,z=0,mass_flag=1):
		"""
		Input:
		argument can either be \nu or mass in Mpc/h
		k in h Mpc^1
		T the tranfer function
		z is the redshift
		mass_flag 1 takes mass as input 0 takes peak height as input
		
		Output:
		returns Tinker2010 bias
		"""	
		delta_c = 1.686
		if mass_flag ==1:
			R = (3/(4.*np.pi)*argument/(self.rho_c_h2_msun_mpc3*self.Omega_matter))**(1/3.)
			PS = cosmology.PS(self,z)
			Delk = 1/(2.*np.pi**2)*PS*self.kbyh**3.
			sigma_square = np.zeros([len(R),1])
			for i in range(0,len(R)):
				wk = cosmology.Wk(self,self.kbyh,R[i])
				sigma_square[i] = np.trapz(Delk*wk**2/self.kbyh,self.kbyh)
			v = delta_c/np.sqrt(sigma_square)
		else:
			v = argument
		delta = 200
		y = np.log10(delta)
		A = 1. + 0.24 * y * np.exp(-(4/y)**4)
		a = 0.44 * y - 0.88
		B = 0.183
		b = 1.5
		C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4/y)**4)
		c = 2.4
		bias = 1- A*v**a/(v**a+delta_c**a)+B*v**b+C*v**c
		return bias.flatten()

	def H1(self,s):
		return s
		
	def H2(self,s):
		return s**2 -1
		
	def H3(self,s):
		return s**3-3*s
		
	def H4(self,s):
		return s**4-6*s**2+3

	def fit(self,v,p):
		"""
		Note that g(z) has already been divided out
		"""
		return (np.log10(v/1.5)**2*p[0] + np.log10(v/1.5)*p[1] + p[2]).flatten()

	def b1avg(self,m,z,fromval,toval=None,mass_flag=1):
		"""
		mass_flag = 0->peakheight
				  = 1->mass 
		m = peakheight is mass_flag=0 or mass in h^-1Msun if mass_flag=1
		z = redshift
		requires mass input in h^-1 Msun
		"""
		if toval==None:
			if fromval==None:
				h1avg = 0
				h2avg = 0
			else:
				h1avg = cosmology.H1(self,fromval)
				h2avg = cosmology.H2(self,fromval)
			_avg = 1
		elif np.exp(-toval**2/2)==0.0:
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			h2avg = (fromval*np.exp(-fromval**2/2))/np.sqrt(2*np.pi)
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		elif np.exp(-fromval**2/2)==0.0:
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			h2avg = (-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		else:
			h1avg = (np.exp(-fromval**2/2)-np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			h2avg = (fromval*np.exp(-fromval**2/2)-toval*np.exp(-toval**2/2))/np.sqrt(2*np.pi)
			_avg = (special.erf(toval/np.sqrt(2))-special.erf(fromval/np.sqrt(2)))/2
		if mass_flag ==1:
			v = cosmology.PeakHeight(self,m,z)
			b1avg = cosmology.T10(self,m,z=z,mass_flag=1) + cosmology.fit(self,v,self.m1['name1'])*h1avg/_avg + 1/2.*cosmology.fit(self,v,self.s1['name1'])*h2avg/_avg
		elif mass_flag==0:
			b1avg = cosmology.T10(self,m,z=z,mass_flag=0) + cosmology.fit(self,m,self.m1['name1'])*h1avg/_avg + 1/2.*cosmology.fit(self,m,self.s1['name1'])*h2avg/_avg
		return b1avg
		



