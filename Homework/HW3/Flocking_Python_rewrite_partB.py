import numpy as np
import matplotlib.pyplot as plt

class flock():
	def flocking_python(self):
		N = 50
		frames = 130
		limit = 200
		L = limit*2
		P = 10 #Spread of initial position (gaussian)
		V = 10 #Spread of initial velocity (gaussian)
		delta = 1 #Time Step
		c1 = .001 #Attraction scaling factor
		c2 = .01 #Repulsion scaling factor
		c3 = 1 #Heading scaling factor
		c4 = .01 #Randomness scaling factor
		vlimit = 1

		# initialize position and velocity
		p = P*np.random.randn(2, N)
		v = V*np.random.randn(2, N)

		# create static agents with repulsion
		p = np.append(p, values=[[-60], [-60]], axis=1)
		v = np.append(v, values=[[0], [0]], axis=1)
		p = np.append(p, values=[[60], [60]], axis=1)
		v = np.append(v, values=[[0], [0]], axis=1)

		# initialize plot
		plt.ion()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(frames):

			# initialize v1, v2, v4
			v1 = np.zeros((2, N+2))
			v2 = np.zeros((2, N+2))
			v4 = np.zeros((2, N))

			# calculate average velocity v3
			v3 = ( (np.sum(v[0,:])/N) + (np.sum(v[1,:])/N) ) * c3

			# limit maximum velocity
			if (np.linalg.norm(v3) > vlimit):
				v3 *= vlimit/np.linalg.norm(v3)

			for n in range(N+2):
				for m in range(N+2):
					if m != n:
						# compute vector r from one agent to the next
						r = p[:,m] - p[:,n]

						if r[0] > L/2:
							r[0] -= L

						elif r[0] < -L/2:
							r[0] += L

						if r[1] > L/2:
							r[1] -= L

						elif r[1] < -L/2:
							r[1] += L

						# compute distance between agent rmag
						rmag = np.sqrt(r[0]**2 + r[1]**2)

						# # compute attraction v1
						# v1[:,n] += c1*r

						# compute attraction v1
						# compute repulsion (non-linear scaling) v2
						if n == N or m == N or n == N+1 or m == N+1: # give static agent 10x the repulsion
							v2[:,n] -= 1000*c2*r/(rmag**2)
						else:
							v1[:,n] += c1*r
							v2[:,n] -= c2*r/(rmag**2)

				if n != N and n != N+1: # don't update velocity of static agents
					# compute random velocity component v4
					v4[:,n] = c4*np.random.randn(2)

					# update velocity
					v[:,n] = v1[:,n] + v2[:,n] + v3 + v4[:,n]

			# update position
			for n in range(N):
				p[:,n] += v[:,n] * delta

			# periodic boundary
			tmp_p = p

			tmp_p[0, p[0,:] > L/2] = tmp_p[0, p[0,:] > (L/2)] - L
			tmp_p[1, p[1,:] > L/2] = tmp_p[1, p[1,:] > (L/2)] - L
			tmp_p[0, p[0,:] < -L/2]  = tmp_p[0, p[0,:] < (-L/2)] + L
			tmp_p[1, p[1,:] < -L/2]  = tmp_p[1, p[1,:] < (-L/2)] + L

			p = tmp_p
			# Can Also be written as:
            # p[p > limit] -= limit * 2
            # p[p < -limit] += limit * 2

			line1, = ax.plot(p[0, 0], p[1, 0])

			# update plot
			ax.clear()
			ax.quiver(p[0,:], p[1,:], v[0,:], v[1,:]) # for drawing velocity arrows
			plt.xlim(-limit, limit)
			plt.ylim(-limit, limit)
			line1.set_data(p[0,:], p[1,:])
			fig.canvas.draw()

flock_py = flock()
flock_py.flocking_python()
