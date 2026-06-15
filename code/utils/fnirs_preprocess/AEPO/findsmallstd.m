function [st] = findsmallstd( d, fs, timwid, timint ,mm)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
% timwid滑动时间窗
% timint 时间间隔
% stee 存储每一段std的值
num=size(d,1);
t1=fs*timwid;
t2=timint*fs;
k=fix(num/t1);
lstAct = size(d,2);
p=0;
while 1
    if (p-1)*t2+t1<num
       p=p+1;
    else
        break;
    end
end

stee=[];
for ii=1:lstAct
    for jj=1:p-1
        stee(jj,ii)=std(d(2+(jj-1)*t2:(jj-1)*t2+t1+1,ii)-d(1+(jj-1)*t2:(jj-1)*t2+t1,ii),0,1);
    end
end

for pp=1:size(d,2)
    steee(:,pp)=sort(stee(:,pp));
end

for ee=1:size(d,2)
    jj=floor(mm*(size(steee,1)));
    st(:,ee)=(sum(steee(1:jj,ee)))/jj;
end


end

