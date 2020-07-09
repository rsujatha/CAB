import numpy as np

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
