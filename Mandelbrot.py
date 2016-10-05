import time, math
import numpy as np
import pyopencl as cl
import pygame

# Default values for pyopencl
#platform = cl.get_platforms()[0]
#device = platform.get_devices()[0]
#ctx = cl.Context([device])

# Manually enter settings each time
ctx = cl.create_some_context(interactive=True)

# Returns array of pixel rgb values for Mandelbrot set
# xMin, xMax, yMin and yMax are values for the actual frame of the set, width and height is the size of the image in pixels
# Higher maxIterations will result in better quality, but will take longer
def mandelbrot(xMin, xMax, yMin, yMax, width, height, maxIterations):

    # Set up pixel values as array
    r1 = np.linspace(xMin, xMax, width, dtype=np.float64)
    r2 = np.linspace(yMin, yMax, height, dtype=np.float64)
    c = r1 + r2[:,None]*1j
    c = np.ravel(c)

    # Set up context
    global ctx
    queue = cl.CommandQueue(ctx)
    output = np.empty(c.shape, dtype=np.uint32)

    # Mandelbrot program
    prg = cl.Program(ctx, """
        #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
        __kernel void mandelbrot(__global double2 *q,
        __global uint *output, ushort const maxiter)
        {
            // Gets current value in array, and initialises variables
            uint gid = get_global_id(0);
            double nreal, real = 0;
            double imag = 0;
            output[gid] = 0;

            // Main Mandelbrot function loop
            for(uint curiter = 0; curiter < maxiter; curiter++) {
                nreal = real*real - imag*imag + q[gid].x;
                imag = 2* real*imag + q[gid].y;
                real = nreal;

                // Called once the escape radius is exceeded
                if (real*real + imag*imag > 4.0f){

                    // Creates a variable nu depending on number of iterations taken to exceed the escape radius, using a smoothing algorithm.
                    float log_zn = log(real*real + imag*imag) / 2.0;
                    int nu = log(log_zn / log(2.0)) / log(2.0);

                    // Assigns a colour ranging from blue to black depending on the variable nu.
                    int iteration = curiter + 1 - nu;
                    int colour = iteration % 255;
                    output[gid] = (colour + (colour*256) + (0*256*256));

                    break;
                }
            }
        }
        """).build()

    # Set up buffers
    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    # Execute program
    prg.mandelbrot(queue, output.shape, None, q_opencl,
                   output_opencl, np.uint16(maxIterations))
        
    cl.enqueue_copy(queue, output, output_opencl).wait()

    # Get output and return it
    output = output.reshape((height, width))
    return output.T


# Set max_iteration value - a higher value results in a higher quality, but takes longer to run
max_iteration = 1000000

# Set initial frame coordinates
x_min = -2.25
x_max = 0.75
y_min =  -1.5
y_max = 1.5

# Set window dimensions
width = 1000
height = 1000

# Initialise pygame
screen = pygame.display.set_mode((width, height))

# Time taken to get data
start = time.time()

# Get Mandelbrot set data
data = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iteration)

# Write data to screen
pygame.surfarray.blit_array(screen, data)
pygame.display.flip()

# Calculate time taken and print pixels / second to screen
end = time.time()
print('Pixels / Second: ' + str(width * height / (end - start) / (width*height)) + 'M')

# Main pygame loop
running = True

while running:
    for event in pygame.event.get():
        # If window exited
        if event.type == pygame.QUIT:
            running = False
        # If button scrolled
        elif event.type == pygame.MOUSEBUTTONUP:

            # Get mouse position to scroll into correct area
            position = pygame.mouse.get_pos()
            x_mouse = position[0]
            y_mouse = position[1]

            # Resize mouse position on screen to position in Mandelbrot set frame
            x = x_min + float(x_mouse*(x_max-x_min)/width)
            y = y_min + float(y_mouse*(y_max-y_min)/height)
        
            frameWidth = x_max - x_min
            frameHeight = y_max - y_min

            # Scroll in - decrease frame size
            if event.button == 4:
                x_min = x - frameWidth/4
                x_max = x + frameWidth/4
                y_min = y - frameHeight/4
                y_max = y + frameHeight/4

            # Scroll out - increase frame size
            elif event.button == 5:
                x_min = x - frameWidth
                x_max = x + frameWidth
                y_min = y - frameHeight
                y_max = y + frameHeight

            # Time taken to get data
            start = time.time()

            # Get Mandelbrot set data
            data = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iteration)

            # Write data to screen
            pygame.surfarray.blit_array(screen, data)
            pygame.display.flip()

            # Calculate time taken and print pixels / second to screen
            end = time.time()
            print('Pixels / Second: ' + str(width * height / (end - start) / (width*height)) + 'M')












                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       

                       
                       
                       
                       
"""
                       
                       @jit
                       def mandelbrot_data(x_min, x_max, y_min, y_max, width, height, max_iteration):
                       data = np.empty((width,height))
                       horizon = 2.0 ** 40
                       log_horizon = np.log(np.log(horizon))/np.log(2)
                       
                       for i in xrange(int(width)):
                       for j in xrange(int(height)):
                       
                       x = x_min + float(i*(x_max-x_min)/width)
                       y = y_min + float(j*(y_max-y_min)/height)
                       
                       a,b = (0.0, 0.0)
                       iteration = 0
                       
                       while (a**2 + b**2 <= 4.0 and iteration < max_iteration):
                       a,b = a**2 - b**2 + x, 2*a*b + y
                       iteration += 1
                       
                       if iteration == max_iteration:
                       data[i,j] = 0
                       else:
                       data[i,j] = iteration*10%255
                       #data[i,j] = iteration - np.log(np.log(math.sqrt(a**2 + b**2)))/np.log(2) + log_horizon
                       
                       return data
                       
                       def mandelbrot_image(x_min, x_max, y_min, y_max, width=200, height=200, maxiter=1024):
                       dpi = 100
                       img_width = dpi * width
                       img_height = dpi * height
                       #x,y,z = mandelbrot_set(xmin,xmax,ymin,ymax,img_width,img_height,maxiter)
                       
                       fig, ax = plt.subplots(figsize=(width, height),dpi=72)
                       ticks = np.arange(0,img_width,3*dpi)
                       x_ticks = x_min + (x_max-x_min)*ticks/img_width
                       plt.xticks(ticks, x_ticks)
                       y_ticks = y_min + (y_max-y_min)*ticks/img_width
                       plt.yticks(ticks, y_ticks)
                       
                       ax.imshow(data, origin='lower')
                       
                       fig.savefig('mandelbrot.png')"""
