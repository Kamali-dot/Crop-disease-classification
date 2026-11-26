function J = resize_nearest(I,newH,newW)

oldH = size(I,1);
oldW = size(I,2);

J = zeros(newH,newW,3,'uint8');

for r = 1:newH

    srcR = floor((r-1)*oldH/newH)+1;

    if srcR > oldH
        srcR = oldH;
    end

    for c = 1:newW

        srcC = floor((c-1)*oldW/newW)+1;

        if srcC > oldW
            srcC = oldW;
        end

        J(r,c,1) = I(srcR,srcC,1);
        J(r,c,2) = I(srcR,srcC,2);
        J(r,c,3) = I(srcR,srcC,3);

    end

end

end