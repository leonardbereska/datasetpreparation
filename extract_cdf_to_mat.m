activities = ["Directions", "Discussion", "Posing", "Waiting", "Greeting", "Walking"];

for activity = activities
%     get all videos for activity
%         activity = "Directions";
%         i = 1;
    filename = activity + '*';
    files = dir(filename);
    
    for i = 1:length(files)
        name = files(i).name;
        
        video_frames = cdfread(name);  % read cdf file
        video_frames = video_frames{1, 1};  % get cell entry
        n_kp = size(video_frames, 2);
        
        
        
        %% extract keypoints
%         n_frames = 0;
%         for frame = 1:size(video_frames, 1)
%             if mod(frame, 10) == 0 % choose only every 10th frame
%                 n_frames = n_frames + 1;  % count how many frames I want
%             end
%         end
%         
%         keypoints = zeros(1, n_kp);
%         kp = 1
%         for frame = 1:size(video_frames, 1)
%             if mod(frame, 10) == 0 % choose only every 10th frame
%                 keypoints(kp, :) = 
%                 kp = kp + 1;
%             end
%         end
        
% %         size(keypoints)
%         size(video_frames)
        
%         keypoints = [keypoints, video_frames(frame, :)];
        
        %% save to mat
        
        save_name = [name(1:end-4),'.mat']; % replace .cdf with .mat
        save(save_name, 'video_frames')        
    end
end
disp('end of file')
