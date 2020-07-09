import numpy as np

class cosmology(object):
	def __init__(self,Omega_matter = 0.3,Omega_lambda = 0.7):
		self.Omega_matter=Omega_matter
		self.Omega_lambda=Omega_lambda
	
	def T10(self,argument,k,Tfn,z=0,mass_flag=1):
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
			PS = cosmology.PS(self,k,z,Tfn)
			Delk = 1/(2.*np.pi**2)*PS*k**3.
			sigma_square = np.zeros([len(R),1])
			for i in range(0,len(R)):
				wk = cosmology.Wk(self,k,R[i])
				sigma_square[i] = np.trapz(Delk*wk**2/k,k)
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






class cab(object):
	def __init__(self):


	def b1avg(self,m,z,k,Tfn,pm1,ps1,fromval,toval,mass_flag=1):
		"""
		mass_flag = 0->peakheight
				  = 1->mass 
		m = peakheight is mass_flag=0 or mass in h^-1Msun if mass_flag=1
		z = redshift
		k = k
		"""
		if fromval==None:
			if toval==None:
				h1avg = 0
				h2avg = 0
			else:
				h1avg = separate_universe.H1(self,toval)
				h2avg = separate_universe.H2(self,toval)
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
			v = separate_universe.PeakHeight(self,m,k,Tfn,z)
			b1avg = separate_universe.T10(self,m,k=k,Tfn=Tfn,z=z,mass_flag=1) + separate_universe.mu1_c200b(self,v,pm1)*h1avg/_avg + 1/2.*separate_universe.Sigma1_c200b(self,v,ps1)*h2avg/_avg
		elif mass_flag==0:
			b1avg = separate_universe.T10(self,m,k=k,Tfn=Tfn,z=z,mass_flag=0) + separate_universe.mu1_c200b(self,m,pm1)*h1avg/_avg + 1/2.*separate_universe.Sigma1_c200b(self,m,ps1)*h2avg/_avg
		return b1avg
