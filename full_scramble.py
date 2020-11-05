import numpy as np
# def full_scramble:
def full_scramble(images,batch_size,scramble_indx):
	M_R = images[:,0,:,:]
	M_G = images[:,1,:,:]
	M_B = images[:,2,:,:]
	#
	M_R_line = M_R.reshape(batch_size,32*32)
	M_G_line = M_G.reshape(batch_size,32*32)
	M_B_line = M_B.reshape(batch_size,32*32)
	#
	M_R_New = M_R_line[:,scramble_indx]
	M_G_New = M_G_line[:,scramble_indx]
	M_B_New = M_B_line[:,scramble_indx]
	#
	images_New = np.empty((batch_size,3,32,32))
	#
	images_New[:,0,:,:] = np.reshape(M_R_New,(batch_size,32,32))
	images_New[:,1,:,:] = np.reshape(M_G_New,(batch_size,32,32))
	images_New[:,2,:,:] = np.reshape(M_B_New,(batch_size,32,32))
	#
	return images_New
