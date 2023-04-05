import os
from bart import bart

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['DEBUG_LEVEL'] = "4"
os.environ['BART_FFTW_WISDOM'] = "./bart_fftw_wisdom"

class bartpy:
    def __init(self):
        return
    
    @staticmethod
    def partialImplementationWarning(fcn_name):
        print(f"bartpy.{fcn_name} Warning: Partial implementation only and not fully tested for options correctness in this python interface!!!")

    @staticmethod
    def optValue(opt_value):
        return f'{"" if opt_value==None else " " + str(opt_value)}'
    
    @staticmethod
    def optCmd(opt_name, opt_value, optionWithValue=None, name_value_separator=" "):
        ropt_value = (name_value_separator + str(opt_value)) if optionWithValue else ""
        return f'{"" if (opt_value == None) else " -" + opt_name + ropt_value}'
    
    @staticmethod
    def optCmdIfTrue(opt_name, opt_value):
        return f'{"" if (opt_value==False) else " -" + opt_name}'

    @staticmethod
    def poissonCmd(Y, Z, y, z, C, v=None, e=None, s_seed=None):
        return f'poisson{bartpy.optCmd("Y",Y,True)}{bartpy.optCmd("Z",Z,True)}{bartpy.optCmd("y",y,True)}{bartpy.optCmd("z",z,True)}{bartpy.optCmd("C",C,True)}{bartpy.optCmd("v",v)}{bartpy.optCmd("e",e)}{bartpy.optCmd("s",s_seed,True)}'

    @staticmethod
    def poisson(Y, Z, y, z, C, v=None, e=None, s_seed=None):
        bartCmd = bartpy.poissonCmd(Y, Z, y, z, C, v, e, s_seed)
        return bart(1, bartCmd)
    
    @staticmethod
    def svdCmd(e=None):
        return f'svd{bartpy.optCmd("e",e)}'
    
    @staticmethod
    def svd(input, e=None):
        bartCmd = bartpy.svdCmd(e)
        return bart(3, bartCmd, input)
    
    @staticmethod
    def calmatCmd(k=None, r=None):
        return f'calmat{bartpy.optCmd("k",k,True)}{bartpy.optCmd("r",r,True)}'
    
    @staticmethod
    def calmat(kspace, k=None, r=None):
        bartCmd = bartpy.calmatCmd(k, r)
        return bart(1, bartCmd, kspace)
    
    @staticmethod
    def caldirCmd(cal_size):
        return f'caldir {cal_size}'

    @staticmethod 
    def caldir(input, cal_size):
        bartCmd = bartpy.caldirCmd(cal_size)
        return bart(1, bartCmd, input)

    @staticmethod
    def ecalibCmd(t=None, c=None, k=None, r=None, m=None, g=None, S=None, W=None, I=None, first=None, P=None, v=None, a=None, d=None):
        return f'ecalib{bartpy.optCmd("t",t,True)}{bartpy.optCmd("c",c,True)}{bartpy.optCmd("k",k,True)}{bartpy.optCmd("r",r,True)}{bartpy.optCmd("m", m,True)}{bartpy.optCmd("g", g)}{bartpy.optCmd("S",S)}{bartpy.optCmd("W",W)}{bartpy.optCmd("I",I)}{bartpy.optCmd("1",first)}{bartpy.optCmd("P",P)}{bartpy.optCmd("v",v,True)}{bartpy.optCmd("a",a)}{bartpy.optCmd("d",d,True)}'

    @staticmethod
    def ecalib(kspace, ev_maps=None, t=None, c=None, k=None, r=None, m=None, g=None, S=None, W=None, I=None, first=None, P=None, v=None, a=None, d=None):
        bartCmd = bartpy.ecalibCmd(t, c, k, r, m, g, S, W, I, first, P, v, a,d)
        if (ev_maps==None):
            return bart(1, bartCmd, kspace)
        else:
            return bart(2, bartCmd, kspace)
    
    @staticmethod
    def pocsenseCmd(i=None, r=None, l=None):
        return f'pocsense{bartpy.optCmd("i",i,True)}{bartpy.optCmd("r",r,True)}{bartpy.optCmd("l",l,True)}'

    @staticmethod
    def pocsense(kspace, sensitivities, i=None, r=None, l=None):
        bartCmd = bartpy.pocsenseCmd(i,r,l)
        return bart(1, bartCmd, kspace, sensitivities)

    @staticmethod
    def picsCmd(l=None, r=None, R=None, c=None, s_step=None, i=None, t=None, n=None, N=None, g=None, G=None, \
                p=None, I=None, b=None, e=None, W=None, d=None, u=None, C=None, f=None, m=None, w=None, S=None, L=None, \
                K=None, B=None, P=None, a=None, M=None, U=None, psf_export=None, psf_import=None, wavelet=None):
        return f'pics{bartpy.optCmd("l",l,True,"")}{bartpy.optCmd("r",r,True)}{bartpy.optCmd("R",R,True)}' \
            f'{bartpy.optCmd("c",c)}{bartpy.optCmd("s",s_step,True)}{bartpy.optCmd("i",i,True)}' \
            f'{bartpy.optCmd("n",n)}{bartpy.optCmd("N",N)}{bartpy.optCmd("g",g)}{bartpy.optCmd("G",G,True)}{bartpy.optCmd("p",p,True)}' \
            f'{bartpy.optCmd("I",I)}{bartpy.optCmd("b",b,True)}{bartpy.optCmd("e",e)}{bartpy.optCmd("W",W,True)}{bartpy.optCmd("d",d,True)}' \
            f'{bartpy.optCmd("u",u,True)}{bartpy.optCmd("C",C,True)}{bartpy.optCmd("f",f,True)}{bartpy.optCmd("m",m)}{bartpy.optCmd("w",w,True)}' \
            f'{bartpy.optCmd("S",S)}{bartpy.optCmd("L",L,True)}{bartpy.optCmd("K",K)}{bartpy.optCmd("B",B,True)}{bartpy.optCmd("P",P,True)}' \
            f'{bartpy.optCmd("a",a)}{bartpy.optCmd("M",M)}{bartpy.optCmd("U",U)}{bartpy.optCmd("-psf_export",psf_export,True)}' \
            f'{bartpy.optCmd("-psf_import",psf_import,True)}{bartpy.optCmd("-wavelet",wavelet,True)}' \
            f'{bartpy.optCmdIfTrue("t", t is not None)}'
    
    @staticmethod
    def pics(kspace, sensitivities, l=None, r=None, R=None, c=None, s_step=None, i=None, t=None, n=None, N=None, g=None, G=None, \
             p=None, I=None, b=None, e=None, W=None, d=None, u=None, C=None, f=None, m=None, w=None, S=None, L=None, \
             K=None, B=None, P=None, a=None, M=None, U=None, psf_export=None, psf_import=None, wavelet=None):
        bartCmd = bartpy.picsCmd(l, r, R, c, s_step, i, t, n, N, g, G, p, I, b, e, W, d, u, C, f, m, w, S, L, K, B, P, a, M, U, psf_export, psf_import, wavelet)
        if (t is None):
            return bart(1, bartCmd, kspace, sensitivities)
        else:
            # Option t with input as k-space trajectory array.  t is appended at the end of the command line
            return bart(1, bartCmd, t, kspace, sensitivities)
    
    @staticmethod
    def fftCmd(bitmask, u=None, i=None, n=None):
        return f'fft{bartpy.optCmd("u",u)}{bartpy.optCmd("i",i)}{bartpy.optCmd("n",n)} {bitmask}'
    
    @staticmethod
    def fft(input, bitmask, u=None, i=None, n=None):
        bartCmd = bartpy.fftCmd(bitmask, u, i, n)
        return bart(1, bartCmd, input)
    
    @staticmethod
    def nufftCmd(a=None, i=None, d=None, t=None, r=None, c=None, l=None, P=None, s=None, g=None, one=None, lowmem=None):
        return f'nufft{bartpy.optCmd("a",a)}{bartpy.optCmd("i",i)}{bartpy.optCmd("d",d,True)}{bartpy.optCmd("t",t)}{bartpy.optCmd("r",r)}{bartpy.optCmd("c",c)}' \
            f'{bartpy.optCmd("l",l,True)}{bartpy.optCmd("P",P)}{bartpy.optCmd("s",s)}{bartpy.optCmd("g",g)}{bartpy.optCmd("1",one)}{bartpy.optCmd("-lowmem",lowmem)}'

    @staticmethod
    def nufft(traj, input, a=None, i=None, d=None, t=None, r=None, c=None, l=None, P=None, s=None, g=None, one=None, lowmem=None):
        bartCmd = bartpy.nufftCmd(a, i, d, t, r, c, l, P, s, g, one, lowmem)
        return bart(1, bartCmd, traj, input)
    
    @staticmethod
    def rssCmd(bitmask):
        return f'rss {bitmask}'
    
    @staticmethod
    def rss(input, bitmask):
        bartCmd = bartpy.rssCmd(bitmask)
        return bart(1, bartCmd, input)                                

    def sliceCmd(dim1=None,pos1=None,dim2=None,pos2=None,dim3=None,pos3=None,dim4=None,pos4=None,dim5=None,pos5=None,dim6=None,pos6=None,dim7=None,pos7=None,dim8=None,pos8=None):
        return f'slice{bartpy.optValue(dim1)}{bartpy.optValue(pos1)}' \
            f'{bartpy.optValue(dim2)}{bartpy.optValue(pos2)}' \
            f'{bartpy.optValue(dim3)}{bartpy.optValue(pos3)}' \
            f'{bartpy.optValue(dim4)}{bartpy.optValue(pos4)}' \
            f'{bartpy.optValue(dim5)}{bartpy.optValue(pos5)}' \
            f'{bartpy.optValue(dim6)}{bartpy.optValue(pos6)}' \
            f'{bartpy.optValue(dim7)}{bartpy.optValue(pos7)}' \
            f'{bartpy.optValue(dim8)}{bartpy.optValue(pos8)}' 
    
    @staticmethod
    def slice(input, dim1=None,pos1=None,dim2=None,pos2=None,dim3=None,pos3=None,dim4=None,pos4=None,dim5=None,pos5=None,dim6=None,pos6=None,dim7=None,pos7=None,dim8=None,pos8=None):
        bartCmd = bartpy.sliceCmd(dim1,pos1,dim2,pos2,dim3,pos3,dim4,pos4,dim5,pos5,dim6,pos6,dim7,pos7,dim8,pos8)
        return bart(1, bartCmd, input)
    
    @staticmethod
    def resizeCmd(c=None,dim1=None,size1=None,dim2=None,size2=None,dim3=None,size3=None,dim4=None,size4=None,dim5=None,size5=None,dim6=None,size6=None,dim7=None,size7=None,dim8=None,size8=None):
        return f'resize{bartpy.optCmd("c",c)}' \
            f'{bartpy.optValue(dim1)}{bartpy.optValue(size1)}' \
            f'{bartpy.optValue(dim2)}{bartpy.optValue(size2)}' \
            f'{bartpy.optValue(dim3)}{bartpy.optValue(size3)}' \
            f'{bartpy.optValue(dim4)}{bartpy.optValue(size4)}' \
            f'{bartpy.optValue(dim5)}{bartpy.optValue(size5)}' \
            f'{bartpy.optValue(dim6)}{bartpy.optValue(size6)}' \
            f'{bartpy.optValue(dim7)}{bartpy.optValue(size7)}' \
            f'{bartpy.optValue(dim8)}{bartpy.optValue(size8)}' 
    
    @staticmethod
    def resize(input, c=None,dim1=None,size1=None,dim2=None,size2=None,dim3=None,size3=None,dim4=None,size4=None,dim5=None,size5=None,dim6=None,size6=None,dim7=None,size7=None,dim8=None,size8=None):
        bartCmd = bartpy.resizeCmd(c,dim1,size1,dim2,size2,dim3,size3,dim4,size4,dim5,size5,dim6,size6,dim7,size7,dim8,size8)
        return bart(1, bartCmd, input)
    
    @staticmethod
    def transpose(input, dim1, dim2):
        bartCmd = f'transpose {dim1} {dim2}'
        return bart(1, bartCmd, input)
    
    @staticmethod
    def scale(input, factor):
        bartCmd = f'scale {factor}'
        return bart(1, bartCmd, input)
    
    @staticmethod
    def flip(input, bitmask):
        bartCmd = f'flip {bitmask}'
        return bart(1, bartCmd, input)
    
    @staticmethod
    def join(input1, input2, dimension, input3=None, input4=None, input5=None, input6=None, input7=None, input8=None, a=None):
        bartCmd = f'join{bartpy.optCmd("a",a)} {dimension}'
        if (input3==None):
            return bart(1, bartCmd, input1, input2)
        elif (input4==None):
            return bart(1, bartCmd, input1, input2, input3)
        elif (input5==None):
            return bart(1, bartCmd, input1, input2, input3, input4)
        elif (input6==None):
            return bart(1, bartCmd, input1, input2, input3, input4, input5)
        elif (input7==None):
            return bart(1, bartCmd, input1, input2, input3, input4, input5, input6)
        elif (input8==None):
            return bart(1, bartCmd, input1, input2, input3, input4, input5, input6, input7)
        else:
            return bart(1, bartCmd, input1, input2, input3, input4, input5, input6, input7, input8)
        
    @staticmethod
    def extractCmd(dim1=None, start1=None, end1=None, dim2=None, start2=None, end2=None, dim3=None, start3=None, end3=None, dim4=None, start4=None, end4=None, dim5=None, start5=None, end5=None, dim6=None, start6=None, end6=None, dim7=None, start7=None, end7=None, dim8=None, start8=None, end8=None):
        return f'extract{bartpy.optValue(dim1)}{bartpy.optValue(start1)}{bartpy.optValue(end1)}' \
            f'{bartpy.optValue(dim2)}{bartpy.optValue(start2)}{bartpy.optValue(end2)}' \
            f'{bartpy.optValue(dim3)}{bartpy.optValue(start3)}{bartpy.optValue(end3)}' \
            f'{bartpy.optValue(dim4)}{bartpy.optValue(start4)}{bartpy.optValue(end4)}' \
            f'{bartpy.optValue(dim5)}{bartpy.optValue(start5)}{bartpy.optValue(end5)}' \
            f'{bartpy.optValue(dim6)}{bartpy.optValue(start6)}{bartpy.optValue(end6)}' \
            f'{bartpy.optValue(dim7)}{bartpy.optValue(start7)}{bartpy.optValue(end7)}' \
            f'{bartpy.optValue(dim8)}{bartpy.optValue(start8)}{bartpy.optValue(end8)}' \
    
    @staticmethod
    def extract(input, dim1=None, start1=None, end1=None, dim2=None, start2=None, end2=None, dim3=None, start3=None, end3=None, dim4=None, start4=None, end4=None, dim5=None, start5=None, end5=None, dim6=None, start6=None, end6=None, dim7=None, start7=None, end7=None, dim8=None, start8=None, end8=None):
        bartCmd = bartpy.extractCmd(dim1,start1,end1,dim2,start2,end2,dim3,start3,end3,dim4,start4,end4,dim5,start5,end5,dim6,start6,end6,dim7,start7,end7,dim8,start8,end8)
        return bart(1, bartCmd, input)
        
    @staticmethod
    def phantomCmd(s=None, S=None, k=None, t=None, G=None, T=None, NIST=None, SONAR=None, N=None, B=None, x=None, g=None, three=None, b=None, r=None, rotational_angle=None, rotational_steps=None):
        return f'phantom{bartpy.optCmd("s",s,True)}{bartpy.optCmd("S",S,True)}{bartpy.optCmd("k",k)}' \
            f'{bartpy.optCmd("G",G)}{bartpy.optCmd("T",T)}{bartpy.optCmd("-NIST",NIST)}' \
            f'{bartpy.optCmd("-SONAR",SONAR)}{bartpy.optCmd("N",N,True)}{bartpy.optCmd("B",B)}' \
            f'{bartpy.optCmd("x",x,True)}{bartpy.optCmd("g",g,True)}{bartpy.optCmd("3",three)}' \
            f'{bartpy.optCmd("b",b)}{bartpy.optCmd("r",r,True)}' \
            f'{bartpy.optCmd("-rotational-angle",rotational_angle,True)}{bartpy.optCmd("-rotational-steps",rotational_steps,True)}' \
            f'{bartpy.optCmdIfTrue("t", t is not None)}'

    @staticmethod
    def phantom(s=None, S=None, k=None, t=None, G=None, T=None, NIST=None, SONAR=None, N=None, B=None, x=None, g=None, three=None, b=None, r=None, rotational_angle=None, rotational_steps=None):
        bartpy.partialImplementationWarning('phantom')
        bartCmd = bartpy.phantomCmd(s, S, k, t, G, T, NIST, SONAR, N, B, x, g, three, b, r, rotational_angle, rotational_steps)
        if (t is None):
            return bart(1, bartCmd)
        else:
            return bart(1, bartCmd, t)

    @staticmethod
    def trajCmd(x=None, y=None, d=None, e=None, a=None, t=None, m=None, l=None, g=None, r=None, G=None, H=None, s=None, D=None, o=None, R=None, q=None, O=None, three=None, c=None, E=None, z=None, C=None):
        return f'traj{bartpy.optCmd("x",x,True)}{bartpy.optCmd("y",y,True)}{bartpy.optCmd("d",d,True)}{bartpy.optCmd("e",e,True)}{bartpy.optCmd("a",a,True)}{bartpy.optCmd("t",t,True)}' \
            f'{bartpy.optCmd("m",m,True)}{bartpy.optCmd("l",l)}{bartpy.optCmd("g",g)}{bartpy.optCmd("r",r)}{bartpy.optCmd("G",G)}{bartpy.optCmd("H",H)}' \
            f'{bartpy.optCmd("s",s,True)}{bartpy.optCmd("D",D)}{bartpy.optCmd("o",o,True)}{bartpy.optCmd("R",R,True)}{bartpy.optCmd("q",q,True)}' \
            f'{bartpy.optCmd("O",O)}{bartpy.optCmd("3",three)}{bartpy.optCmd("c",c)}{bartpy.optCmd("E",E)}' \
            f'{bartpy.optCmd("z",z,True)}{bartpy.optCmd("C",C)}'
    
    @staticmethod
    def traj(x=None, y=None, d=None, e=None, a=None, t=None, m=None, l=None, g=None, r=None, G=None, H=None, s=None, D=None, o=None, R=None, q=None, O=None, three=None, c=None, E=None, z=None, C=None):
        bartpy.partialImplementationWarning('traj')
        bartCmd = bartpy.trajCmd(x, y, d, e, a, t, m, l, g, r, G, H, s, D, o, R, q, O, three, c, E, z, C)
        return bart(1, bartCmd)
    
    @staticmethod
    def circshift(input, dim, shift):
        bartCmd = f'circshift {dim} {shift}'
        return bart(1, bartCmd, input)