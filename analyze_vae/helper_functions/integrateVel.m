function [x] = integrateVel(x_init, vel, dt)

    x = zeros(size(vel,1),size(x_init,2));
    x(1,:) = x_init;
    
    for i_step = 2:size(x,1)
        x(i_step,:) = x(i_step-1,:) + vel(i_step-1,:)*dt;
    end


end