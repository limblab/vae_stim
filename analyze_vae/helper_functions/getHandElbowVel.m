function [hand_vel,elbow_vel] = getHandElbowVel(point_kin)

    hand_vel = zeros(numel(point_kin),size(point_kin{1}.hand_vel,1),3);
    elbow_vel = zeros(size(hand_vel));
    
    for i = 1:numel(point_kin)
        hand_vel(i,:,:) = [point_kin{i}.hand_vel.X, point_kin{i}.hand_vel.Y, point_kin{i}.hand_vel.Z];
        elbow_vel(i,:,:) = [point_kin{i}.elbow_vel.X, point_kin{i}.elbow_vel.Y, point_kin{i}.elbow_vel.Z];
    end
    




end