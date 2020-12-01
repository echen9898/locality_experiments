import numpy as np

def full_scramble(images,batch_size,dim=28,channels=1):
	scramble_indx = np.arange(dim*dim)
	np.random.shuffle(scramble_indx)
	if channels == 1:
		M_Grey = images[:,0,:,:]
		M_Grey_line = M_Grey.reshape(batch_size,dim*dim)
		M_Grey_New = M_Grey_line[:,scramble_indx]
		new_images = np.empty((batch_size,channels,dim,dim))
		new_images[:,0,:,:] = np.reshape(M_Grey_New,(batch_size,dim,dim))
	elif channels == 3:
		M_R = images[:,0,:,:]
		M_G = images[:,1,:,:]
		M_B = images[:,2,:,:]
		M_R_line = M_R.reshape(batch_size,dim*dim)
		M_G_line = M_G.reshape(batch_size,dim*dim)
		M_B_line = M_B.reshape(batch_size,dim*dim)
		M_R_New = M_R_line[:,scramble_indx]
		M_G_New = M_G_line[:,scramble_indx]
		M_B_New = M_B_line[:,scramble_indx]
		new_images = np.empty((batch_size,channels,dim,dim))
		new_images[:,0,:,:] = np.reshape(M_R_New,(batch_size,dim,dim))
		new_images[:,1,:,:] = np.reshape(M_G_New,(batch_size,dim,dim))
		new_images[:,2,:,:] = np.reshape(M_B_New,(batch_size,dim,dim))
	
	return new_images
