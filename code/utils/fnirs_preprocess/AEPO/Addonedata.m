function [ dod ] = Addonedata( dot )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
ww=size(dot,1);
www=size(dot,2);
dod=dot;
for r =1:www
    
    dod(ww+1,r)=dod(ww,r);
end

end

