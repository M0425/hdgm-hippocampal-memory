#!/usr/bin/env python3
"""HDGM V4 semantically aligned causal perturbation experiment (V2).

This evaluation-only experiment uses the locked V4 primary checkpoints and
never retrains or selects a model. The exact full V4 main-experiment source is
embedded in this single file. V2 separates two causal questions that V1
conflated: whether a module-specific computation is necessary, and whether the
learned gate must remain aligned with the correct trial.

Primary perturbation: operation_cut
-----------------------------------
The intervention is matched to the actual equation of each module:

* EC: remove selected afferent input and its memory-write strength.
* CA3: remove only the associative recurrent term M @ h while preserving the
  learned CA3 gate, local 0.5*h term, and five recurrent iterations.
* CA1: remove only the memory-derived term M @ g while preserving the
  predictive 0.5*g term, learned CA1 gain, residual path, and feedback.

Secondary perturbation: gate_trial_shuffle
------------------------------------------
At the targeted module and stage, the learned gate is cyclically permuted
across trials within the same batch and time step. This preserves gate range,
dimensionality, and population distribution while breaking alignment between
the gate and the trial whose state it controls.

Stages
------
encoding    : Cue A and Cue B samples.
maintenance : Delay 1 and Delay 2, excluding the terminal readout window.
readout     : the final ``readout_width`` recurrent samples ending at decision.

The default output files are:

* output/figure1d_data/hdgm_semantic_causal_perturbation_v2.json
* output/figure1d_data/hdgm_semantic_causal_perturbation_v2.csv
* output/figure1d_data/hdgm_semantic_causal_perturbation_v2.npz

Run
---
Run this standalone file directly from PyCharm or with::

    python HDGM_v4_semantic_causal_perturbation_v2.py

The script restores the embedded locked V4 main code and automatically searches
for all ten primary_150 checkpoints. If they are elsewhere, a folder-selection
window opens so their existing directory can be selected without moving files.
CUDA is chosen when available. Command-line overrides remain optional.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import sys
import tempfile
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

try:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for the frozen representation analyses. "
        "Install it with: pip install scikit-learn"
    ) from exc


FINAL_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 111)
MODULES = ("ec", "ca3", "ca1")
STAGES = ("encoding", "maintenance", "readout")
PERTURBATION_MODES = ("operation_cut", "gate_trial_shuffle")
CA1_PREDICTIVE_CUT_MODE = "ca1_predictive_cut"
N_CLASSES = 4
SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDED_V4_SHA256 = "cd27093dc8c8d9cbb99085a880d48122a8853703dc7e054d0334bd0acedd9d8c"
EMBEDDED_V4_SOURCE_B85 = (
    "c-rl~U5_J4k|6e;zaox}Lz7)h7Rk!2?#^uvwpd-=)3~mlQdRYAFLDWXBpJ+%E`DtUnU&c!`F6mqfE6U@AnEON5DpL|K&un(;UH;G"
    "5Be`Nb9%bJAp7tS^N$EJGppz8X0Y4E3^zA-H#avoH#awHG#Up#|NP76$uEyilKE=#CYvTjy3VJE>Ev&>d6{qW)iTMh)>)A+vgM|8"
    "aPayfOJ>{oJo)9LWU-oN^CVqPlTBK_N%F;do*`ID)zTtMifohS%WT?7u-;^~oaMzL-K>ghX!4-gE|*!6YzlxWli8|BddJ;lyPl?-"
    "tZXOQWjf#DPv}89*(B>Cn`V<a;GHIui)`{{y~>vx0Nj=UlOD{9th@lc$wfAw9<H{Vq+HMQO`CeS%r2p+GMi_U4FIdYTQtepbplOo"
    "l2vg4?dDS+1dJT|wMdtftdl%Rj=SAM?jH<oox#7;Lm<zp+!i#vO|hCMIpDuc^LctU&koL@8`HeR-&1U3n$6Pf94OfU+8+SPR>cNt"
    "uCMj)AIjBI|6Y~)R{>*KEntYNMFP#IK!y@Wt%}ofnonS)rpTZMYOm+3%{)Ilus=IMv1a4R`FW!i!gbcy_zQ3*>-k2PFSm>JHGW&J"
    "^(UCd$%XSrXSpPhrTcZY-sB4`pdSYZA0&gj{txPQJIOFD^9fAmX1l%@vT1giPqJj7`pJJb8<XucZ6x_jea1gKc{w(d+H57+yv&lu"
    "WW5Cn9sr&h_GWxOU!A4%ahYY)CjRFn0nTcj93=2Rp23bOYth%`x)U$prh)Oc5+Rq>^^6;{7ziYerW$CR&gTlyZSX7w8L^y}SYx@|"
    "(X<CV%{E!F$d`G!fyo;ruZwL~U*jxWPA(Q{@dlv3NP*%9_vM{lCwXS18Ox*dq`6Mv6Rd!InShWeVcoax)y`kqWk~-OZ9b(YnNq3="
    "=mrRna#L(48x$jq0|8)_j?aKSCKux(OQ#^wPLi_~NEk&t2TQuW%IA3slJx>$VXe((z<tT)A}jOqFdU-S&SaZC83T?n);7&B)dkFO"
    "7tH!1h5uFnIURnJW$R?MoL?u%6pM78EV9W(3jDEvy@;fn+r@06umefnj1j~ohv~wKH!kXh0~Q$$u-AFMtieW5AnM<Y-xm2Miy?G|"
    "b&i_@0wr&={QP1gM}cK(3|x4U%;u|f1DLv<Zr8%du0<d}v5+oNLfBqkw7e>bCOg}HvLD!R4=~FQd!S=7$KV_izkc>S(UXjPPrPoa"
    "yvY{zLv@fXSJZ(+;zVAqMaI?g<vA{)+O@--$7-rAU=G-p=D4dgd!24OasJbC8z<EVo3RK#Kn=#xl-9*+vzn~tC%BlCvwSsQo#zu+"
    "FBjMA6|ZlsFv$wgqDXyHh=dZ?hH>JSe!W<4Rw#c-W25~z|B+OkPV#-WUQI5_?@@u#iW_QW3L6UNw`l>o*J`nbDrX>WHrIbaoq{ea"
    "WsY>Q@7EyhKopIYF86(ha4>$_6oERR>(SS<bOQpe%zw;u$s^|3d12@6advp*k`9{;R8>tXK~JwMe(Qu|EYwLkeK2SJ<2Go0?c`(l"
    "x7Wo<JnptcO=$5ftEi9l+&ZGK<IM%?tE>4`R_S)IQn&M1RmiS3upz>Z>~;TGtsPT-yAiaLNB9e{cnm32-^FT~1OK_)3FTv&FhR+#"
    "TjEUR<0Dg8;ywqO1xN(B9zQ;&=BX4uQ29qo)Vb%$0~o;}eQA9(UoFoKE<m0Y@Wbl^%=K~;6drpkewJSWW1B;QI%c@bnpUA81}q6n"
    "50VwAN#LnO5T1@a;*`=+s9xi_LQ#!i&NEmlcz8%~*(Rjbr_cz@<Qk_Fy8ZH-FHtAR(ght~pq4_$2N_sq(`34(*)c){4p4lo;bRUi"
    "Y9$cMjX!IO{cTZf+R|k@zXtIVRF0I!>eKm)HA*EpH2{}e+Ks0<tacbMlr>1;j(L4u9*tp=*4s__P*A$z&1kXuF<ZjfYLa6khvf!0"
    "VOpAwB*3Lg)ba<^?+S<rj~`0}5o$X1E6Ksdt`zKdWmPHidf8!L6!GpdhB2ns4v;4ov*l%8td^wUiaoIa={=rjW&%FpW>;q=kbYr$"
    "qvmA`?d8+Wg*RQO831jI4Qd<bnGiIOK&%|G$N~mTuX_Q+Cm@Ns0-_&4kg!rnmZ%U|A~C;#dRvmfbrc*P$a0l~o>6Y5o}$Es=^``S"
    "HCt^9-msa&q)fg%@;K`<FX2$6wn#zGGQG_9;MoT66d4GGk_{tJ&72Hz{(AR5)AT_S**U)#GFE?HU7+xKk!~)&08+t;WdpkmG;yLP"
    "Z<dTf#=^oLH_iDCB+jgDPji~v2A_28SH)|C{*nD=IBnp5bogwb*vmpl9vdju@=E)<RI!yUX5Jd~?H9{g17s)xs<~=_%+wac&8CBI"
    "R?Cb(iK8zPqGQIN3@j~N5Je0CP`}(_Ll*!rzXK45N(||Iaw|xp0kHft0@!x}`!PVKB!KV_x6l%mwsrl7hd~n<k?7wCE=++Mh_IBr"
    "q<>zajAtVS+9&V_?$W5{;>56p27h_j@MN$^*MO1!#9>XBO*2bvJs+)3T5hh_SrZy2iSnp#%_eB8%<*hMM6tC6BPhFxZKT;sK9e%c"
    "fyDp);PeUPLlZdHq7LBYa8C1)uC2fPFgl1ZOgMC&yZy$k@eJ>X$Kw#vQPeW%xn)3MJZy{^2BrCd6+fCjBz54h{kEN4!0t#ZQcQs+"
    "KQa)o29~;1hf~hBSPItChz&|RbOoys#z5T;E>-tFDlcGfxOcafO@*koM;6NM4E5#SeQ4G<-W07)c-t5k+j*u4f&54R9g2=9JV;3_"
    "Pd-Zep!O4x7($ui-NKj@^06_M`GZ=dSD@bIB$4z1Nw3ziU9ASpTdip~Sb$MMd)ghf<0AO9H-gUh+@;ELHjY}#V2jrIT8Xk>gR&n*"
    "+4oR7IB6_DqT~csECtB}<apq1Ms6?8x_ZmOYK)b;?js%=#Akr`Y!_m!$D<X8n;7~{^bdSN>s0#8+lND8tN6}zzipdoJGk0vOzogy"
    "I7q-2^K!e~<csWCQLKt)<2V1qzy8g?`|tk0fAKGV^RNH?umA4f{`$Z9JNOfXcn+u8-~QkK@;CqLpZ}+S{QLj-zyJGR|4;ww*Z<~!"
    "w$eTM&ENg^zx~Jm<=6lAzvgC`RU2Xrs3kzCs`M^k^a<JmU{FFkaZ83?iJ`!F1!uG(pJpMzuogg4s~Uz_x~{ZeRk}#aH)GQXY%c>c"
    "y9TNk$&bhuDpVCx6$F*E>|<>K;YDG~A;Il159=@w#nOXX;S89{Ah<&E%hD+A;}V_Lh%*ZY5n33$If&zM7-sJ!V@Ova#@*GA1PfdJ"
    "iG;Hwpb@D@qE-djTs8CkNp?rNoBs48{psJKKX=#`?zT1DVtcsT7IC+2BDPihY_M~$_^G)*<akMOvX3-FA<?SbWlfD?T@8<X%xn>9"
    "QlZUQa(73`p)=8OUBzkd(3vhfOX$^r4iS=VK6BcNfDQ$yY{dtyRS|3-=IL2B*R$v7u|k8j;8!s7a7uyy5);sc7t}t5HD>?+z)?mw"
    "RvC3W;c28-{+Vtw>Y%Sk(_5tZa{R<uY%nFC0Ut2w;`0J^hW<3#I>V|unuf?`$-te}Kqtd~h06&?btTp}9ZFB}$0&d$YIf2!n#G$v"
    "f1}_ghP#^>>L!Lens^j%LXYyKzV`7xI~h7tI1|$y#YK*Tx*N(r>YTe${Q27pb-x<*&_^w&dk@q{G1M`l#K!A$Q=kWu-cma|S8O=?"
    "F)LsJ&GR=|^9sZ>nk5ENvqZo<;w0axF>JuG)iq~%C`&!8E0-WBH*`_N?BN0f$u;ao^jV!`nhq@VEiH=lx{1;$7;rld%Tmlh!;bLy"
    "s0x-hy!}9NUUvYKB?aYf<U8r&(a%Hgv$s`Nl@4R2{{9y&#=WnqK$0~)x6!3g)w#{y7!4lvqsO_j`aD-w9q7uPC%W=hN4m29Ojp()"
    ">dHHw>YVLW@2_RJ!8*Hl$<`+RXXrvl2qkp7(pBo{TBxEvk6-J3eG7F^Whb`t>}~Y&NIGCe-aV+6?s-y{oaUnnT*D}?VI{6%fGct|"
    "wJ0Nu=3<TjL>d^@H89-Oz;H(cmcv5)aV~e8h{J@fxF<x}hPz07>~3ikZ@&_6zZP%b#p~`!u&7`y;c{vx*U~oIPJV=6Kf=FL_&+?&"
    "n3GK01QjzriQnxSz@Y#fwyV#bR!oJT+SR8?r`m8=wYV88;GcGd{XkZ6{;sId_bT*jDul8!_~KS--9i6I18u_;t-T@=!KTnVnzI1b"
    "rPskR^@%BSjcj`AWbv>S7e@x>$0rrXU<)y<KK8P7*O1fLN!OM}ZSc0Zze^=!7w1j3ELR2Uj^zdpeW0JUv}(ivaUCsQ8K=*yj&4$W"
    ")y4U}b6sM9+t+4&kQwBzwTXl7QF*vWWuX~Va94W=Pwmqg#b9?Gi(<g~>aBabmptkuPc~?v0Q~oSY%xnxjj8#(oxsjhZnAYbI9A$_"
    "cTY!!LQ%Rf4>dOF@&a8Y;n;NnCpGnWV6h;qo$a#x+ijNpn8C@m)e*lB#M->LH(`!C$xHUiFW-aH&0poq>FO;|I$I(Gy;!a0&E>N5"
    "d^O$9wYKQE-te93MLwNobcHT`P`2x=Xtp{A*1N#hzLtRh>wKA|Mbn~b8w)VDrjDbkvA5L6@hnR>a$X2sX_N%~tLcE6IXDzD)Og3M"
    "=_^;W%_6;OGB#l9d@&G4k2Me^Lobo0swk!z&;loTO>P%W4~3d}#?;mw06yRu7}84%HKhUV8;!;*B<p34o|oArOw(aH&(G0E8kYZR"
    "niU<U+xOpF@$volWTT)@L;DHedu-S1c}^an^J^eMvbg{aj$g>6b2R~s7T?ZYE5FYSeOt-p`t`Ht59#e*kxjNmf$o0O>t(vgCna7C"
    "p<g<C2>&fRx0apcDf}1y1kgFCe24@ny`7VvJJBGY<rDUFV-{I#(Q!4?PHd|JT~^aFHhjd9PM|00<XStM(?h}QB;9US%hh7FWrOTl"
    "c9CA@t0J8nxoK*4G_7~@dmEjO_!ZpX=oZXRle6aM8?i+vR#e38WzUL{Cix)`uew(}3xOIPUyuTE+XsJ{ef@1S{9KDAdH68t??x3H"
    "{vZ1w%trmkK9Jc~8R7Tpf75Rc#O0&+H59W9TQT#OyMrvUMf(<Zm$s7GaMIbN`COTZ9PgoZIZny&;(j}ee4D!8_?^>cqs%+5bxeFW"
    "1#l#Oy?_Jp7{0?mI^<~O+Ir5c_A(5-i{QxJleaYG>DkD0sA(r>&fQu7f@8p45+GrtXoe(%3dv#8qpsqQR?N)@9g@{NyfqhHS}rzW"
    "Y^P^<$$OFlo|94Jgk=y69Rx!M!CCj@EP}xF8Idq45qgS{2c8ca&aONyQk_-^bSBS?RX|g7R+Bx5oG@wfoo3%gXy8K)5h~<NrQOiE"
    "i9ZIqjp5h;_ACxNjloV0tYyj6nOzoosu#kU^~{+<33Vo*rW(o<qN;V-EWJ+AT?CDyc{+np&tNpO>4>{2G(mjMwlVr6OSUHAaQ$Rl"
    "SrBzsd>5Eb8jQVhqM8YfD-AQ}0A%Ms4mcEm!#Y4_RRC0TUV%*lokf5gR@H!07x2c{aCxLgv3*5>8LAZ9pG3)8W}DmZt^THjzMyyC"
    "n6UOA!4~IjX6{z>r?ul)S#P$M&LT^f{yELrWo#w0om5u3-Ogg}QmvG7H%GnAIcP^(%6hUsB9SAq+xkqy3L&8T8)%L@$#Z=5_dcu;"
    "aH#w`-(>p;33VIpN(=Ff!aDa4dgiZeaYkPr^&QCq<er`T-ut^`%#If{LJ|x_A8-9o2YQC<1OAIQQnT~W+s=>f(DcuLv&s0%F}Fl$"
    "*NQ%k0D5u7XN$9Jiihl`rK?ZXR?uqdwCY0FG!8Y-E6qdI$6RHvsH~~|Ck}#F2td|OB-q74@Y+R{KohRDCca}u#CGU?;^B2S%bj?`"
    "w?!ECqLKuW*F0R2q9oK-ee5?jPZwv?biAOKj-U4Z$5()oN*?vYovOkO;Yz%Si}@2&x)IcP5AYu00mM#q;&I)Ze64w)!YKeZM&1^$"
    ")TXFQ&W#`@r=RAq?TZ5x=osf`aHRtN*tvMVN9Envtj)5>rW7xPAER2=hioyP=jEoITwSCa_48c)i#)^%oc=((AwC7Z548@4uW3*A"
    "{0hg2<m&VUIMUtiQz}VSiC2lGkubp<>DB~{E$?bkfg4%W^aNWo4eFeak!0~U&YS10<N>4gG>kQOTHY;CXYKuh+X?`ymLAB}@)08P"
    "F<;PA;^5lh+`o1>-*N3=F|wQ?%szp+F)2)&1RvbYj@>hpmOD{s!9E7wh#OtQts?l={(g`&o7}}i#X9INK58W&p`Ol4nTH5a(|%wn"
    "1r&A&)lS?mJ&#<^b(f8_`v|^0V9ah~dR3Ww>5vv?xtZ&59FNa~65zaw^lnq;b2aVV3Zg*J42Iz8kyccAhvqBEnCdFo3KI|be0;%j"
    "LxP{jVZ{(5l>+JH*OnuW)%Pi4FgQwq*a2M0W%!t0_PQ!Pfve@^6q7fryXUa$XqhjeTg!=Z!&WHTu5!^jscA>nYu?woiE*b|@!^VA"
    "Pqj&7RHN0n>-~-$Z0#cOzSbG85;4vg>C!l{lBRtZBMuA=igAEp87>Sc#D5~Cyf~noe?%a>C4qf)zr^bJ3ju5S->U35hNPC_;?xnk"
    "ITKe{GuFsw7*SMNcUWC_xT~%*cE#J}qMCLq>kez{x~8$9zE>Qx76M#wIc%8Fx1!-)YwT9|=lOcQnxu<$I!`|53)tk#l<rjNSk9q("
    "I1--<C1wMvmYwKbXN7*QkB!zN!m6QR><yvJ%cHpeRUFA>gbZ^u7iMCD^UeUhLAN7nPWl53b!I)AK^$OA4FRD}`a$4~KaAr(kJ7<G"
    "?z8;ab(}qm#O?UAwcsbS7Em=6^e{#sPz%MRJ&^M$(1qA0>7$BX1UBSo?Osi7zZ=z5Jj19UKU9)KVEMBF<=zGqpKfmS({OlBdHv1C"
    "UGMyBE+s0jI1o)c@!oIkoGX7Yfhy;dX1Iei!#^d`468|VOS7B1KrUAF{gWf#If$-byy!O*^tqpSC~z-@&-l-uO?tmulb*(QS%FQ9"
    "_xEZGjK4a{CJG3)E%U2aQ-Q-4V=T$#`Q}1ZxpXTWR2VqwtdBpv#nJBd%7l=gFIKsFDX9wO^&K<<w1dGJ9|O;tbTwYAa9L}_A7Q_4"
    "W8{SD{`KbX(9VheYScmZsucKu?40i1am1nexD_59g$7hbWGIzXz5p6m5tt>xWLvx!S8adn7*82^OLwZJaGD`}#BGNUf;5GURZnAx"
    "*^w{)aE!(07^;*N%<9SM$;mlyb2n-hfwn+%ts<Lmo80q{lBV8~9+++DBh!f%klyS=tv_|s^`)_$!N&a&{Q1Dq)1!^MlGe%@>#)#k"
    "Q`md_85;Y3I%~PxvA}cw_hd|0JDIa6vq>uuZW2h^lFhQ#j4tTh^3n9XwiHj-%%4-iqhQqpTsoC5|FgS1fGLdt8<oX_#tDpoTg4`T"
    "@5SMJad@7zNR6mU<W;_mRlNN3oyni?Oy1L-i8+SH=f`<q@0byB8X^bz${PY>g|E;MYg`ETbR%?I(Q>NJ*{@aILuYB5g20(BPQs8b"
    "0!Qn>bMoIo{NWDb5BDSf@E*k9!$H$+#Up;yX|h$_Ly!2wdg2e`#2-4u9|pw#)4nZKdmw<QUoqcrwez7Px%wG}>RG5>6_nlO0M7mq"
    "fxOgEm%rzmPT_WM_{TWkn=h3f_+Fi1Wt~6YsQzKzsNT=v`LRv_xX3mI{Bx1b*I7~C`*AwvHG-4SSU*N>Dwa|aEtGV|@mNWj7U$?3"
    "uOWqIr-o1A9GXp4YoKv)n1`l5dv;*eE}AW!OHlU&Q%Z2Eii$qP-e>IJL4<mDW1ikm>{etiXil7fO=yJSmT@fNtfs=V(X(qk8U*S3"
    "iePy<8HKuLnN`3^@||J}BTGd7yONY*5&)e?Sxm4+4~@o)6qxAhJ<@fu@QlO;mB1Pa+F<748jkrm6+h-{MQqHSm`*NWaFg}6iDm%o"
    "i}zJVLSCF=Sh8z?#DCYVQSv}X*w=xt7adCJ+4Om?%oq7Qg>B>-bqMR{@EoX%iPjFMd3wHDrgNwPXMrrS74T}}ZI&$)&cAb+p*Lo!"
    "p2cGVSe2|yXHny>WcWUw=FVbLvw|k4-uK_PleZW7<RY1_RE`Oh+eNlmT@uPf^r$GHQG9a_YT6=w11!$Z+|}E4O!br{7nsFtvpP`W"
    "+p+@4xxr`ldFdyy&8{Z%ErwT0=ag1ycDPRS0yEOC=lLYZrl4<g=H>KqmD59b;4Y5yiYQjIB!z8s!#8430fv7Kd{&%i$v4Rr{67!Y"
    "efy3JDuX#9XM{oWNUTHk7zne6rE9d|wC&@d)5D&6Zs{&6Rmq@M&=~VG=?BsHc||pr>9V2H@vMpkhf<tlc!Ki|erTdk!Bt)gtCl*C"
    "p>-`ge0*S9P972p#)ThD`<zW@yI7Dtd_e+)kSc8xIBu-xcMOF_CO}`Nt4))uwiOCw7>rT_NYM7RaK%7?>!HSlw$G)0yyPy2bqdVN"
    "i=myMH!+Otk<RHV$p!5a7h`~yvyv7M5Eq~L_&7m8tug+=nAXV{lU`z4f?%})n$;XCBp@sg(bA<B7FYNSOniZf2>}$dlTi{;%*5J>"
    "@z6hUc!)#S{}NY&3S3{?H}Y7*yve70OZW3pqqUO&I6?Vr#)M-&X|+Q7LYFC#pgaDdN;Y_?lMT)@&YkiXDEnqG?C-y??eh2E8!~8#"
    "x0Cw!)=WO`Bwusp(uXEH+Iv?|{IsP_q4Osy)+QHY(jtB5!x}jVr6t~{(O+l?KLL5s*{lQ(=gw2LMH|0^0k}yeZv>$TyFdj{SLXYQ"
    "gQ)v(o$*zlk-f9A=%1DxV!wNf7jPCto{(t-AodaU+p|+MAb=T(_?v1J7SI`(1FS}4LB;S0Ft8#ZDDJk|9F(G;C0*K`oHk{tZ<H!U"
    "tTl0G!&3}0t7+R{4G6xu$oaFp2deFv7~uQvmqcxQrG?czTQ(W5uRE$G>;%H<VoFtY;a<VMPBk`aDE#Km$5~oVGF{ln_gG*3d#x|Y"
    "_3xGK*B9<e!!p<MX!#dmK|8EmGhF_tcDXQ2*AI>wK=f<Y5knk}LZ=fS=6Dn)(D>;W*^~FV$o>G9|C5^K-}^}@CQ<*P3#C12B{=l`"
    "Q%fShHND<r&xU*U>?aZAx02*6^5`lV)jPnGPqYLt^K;P2bOdQfZexN_mD0}Iox3MVL$9QUH7xc{*mfyC^-nNT2lBS~_=w?#;?t3a"
    ">x+-aCpOFnE39mi!Tx&|ntXcIrevm+mfYz%nLWy8Xs^-nvIkh(I^-ep(^=yk*1s7e4I9?pBI(6e7Cp(Xp@+DQyxO%(n+|b+=<lsg"
    "f6_^wZi^zHZ0B2a=%(=5?@gap(TvM<mW^prY`ElRw!CbMsO)XZt2}6MTIz-(GUkz^v_t?M%Qj?F=~@_($dl60tu}BF;f!AlW1lF%"
    "X%zlnHFB7sD-IzrtIS968Sq!l3D08w^}qRFfBQfFZ~y5Z|DLmz{rVsN&)@!E|L)iS;a~pxzx=PTiR3^2!~gT!|L_mL{%`+Y<s#$1"
    "Z89<nTWVEx8UFkr7RWOi_8bSW!jRU3q)}e1icQ0nOw%4{UvE|Lk$#(q<0VxBs(1+|rg%$oexYh96b~>s3C1pop?H99!agMrF+^IJ"
    "b+6~7A=T-XxRthUyX8k3S*YyXgGe(|Hy6^hUoqys9JBYF87k>9%Q5!7F`ACh$UF?TNdbmY#ue_*2D)#w+9O2hD<brLB9J42nQ63i"
    "R!($mih85?^wjbS|6R!%@pgbLpH#C#$O)=Zq*BjPsaH#-pMq6lla5rgGg-@H{P(R`iW7x(ck`5L>wPhje@kGdJ+iQGH}|)h_utL5"
    "M%MA|rs(&#;=RD>Od+ASdXdu^S<6-QB0;i)z}HIUJKN^->G&+2qf^pUC0g+|a0)n7m2reSG>&cYDsh-al>F1=$QriH{Tx2aurLoF"
    "YO9u$!9&`%Y%UJR;>9^Bm&qwCuAa!<q{@b|GQXWpcC|seaz-t5RMx$0RpkkB>I}U)tZq!N9M*^O#g^|#<n|sLw3#>gy>$v0qtu9i"
    "4>kBfB;A6^zanyB3~r;`4FWyS@`KBixXR+`eB#*RiG1WA6$yRhA2s=Q0!mcp-tj-w<>3jw9Oy{$cDYTAUS6}{vh`|mF~B6|B=YFb"
    "a&X-3nj6*Cd6CiTp|a#5$)uiJfv#nE8UpFtv}aC(>@$3QCGX2f_s~ivd4Ukj(J_m*re5>5=u>s7VRc2{r}>RMe!RPpesv=sM;mc2"
    "?JAJ=s*&nJX`}~+J5faKBPr!-5Q33cgIEj%#|!4(2fa@_$rt$*-_Pb#k}3~3S-E+CcPDeq{Q^TB!_pd;>v_K6ftOh}wYN@B5iHr@"
    "!MTK8af!wQzTZvG^Gn>q@iv*NCn+W^pJup)VeIvn-+Y;%Lvgw=_f7<djpD1=i?qn5FVfA$7praYl#@z6TV6K9cl!h97#O_%r{W+I"
    "fH_(YoP(zK<WHPW2eCtIAZskB{1C_)53KKlaf)OD;Gg1EB3|m@lZrDcwttG+I;kEb(0_^7g)oT&{EvE6NdNr+55EpGQ8fE)$vy)m"
    "T2ba%O8y6cH!oI8y!ICp-_%aM3Ll!)@OV7U>0&>n6!I$8{W94uOLTo;;WWP{wF(rRDX~+>WeA<mQ|Wo?FP+-c1=R^MrJbP;c<2HT"
    "tAPtrNv%MAN&mdUThS)13&*0Pt2x`i3|B1+68P`(4YM<@EWYaJB0b;!)@a2G|81wRGC7=Gt6yfztiapw6uP)A@!}iB#q<z;`>-#`"
    "+kA6@f6xstE#5*Y;=EX)a&)cM3X6fI%=vr;vUI#$^G&it%kyl*Ulr9i#kG~|A0(fX_D{M5B~C!jLT&4CRbWhn@?txiQHhEhGSAM_"
    "$+Z&XMEa}{=#Z(u*)CpOe^{zs;Sf8?3)ri&;*!0V;r9le1VDC`t0Y5L81`|Qq)Xsofi9ivBopQKOl6F)#DgC{kY|s|)<6?&29D$*"
    "?twGM$Dj5-Z3!J|iT21-HvnyHOA5Rb&Fq9Op*u9MTWC2IdAi1)O4zP=Q68~-8TPkOO65f-nz`TzNk7dv9(#G4@A8dWh-xA;>fC(1"
    "QjIq0PqnkFRil!=nJmR^Ct0FS|8|?l7?Bi+)F3xEW)nU=>E5iG^5QeKWvO4row5MR2Tni{*jv~r3{wtB5t(=NhXbuY%{xdSwF*%p"
    "s5^w^YJ6U#Q)w_~zjLb8^j<e|!)@>3Z|tjhTE9Ds)L{RFQiEMNrXufR{Ol2eL`_LO_mpEVKpN>}f-cs+U##<>rtQv`h|kb)+*bvn"
    "h+AtNO4k`v;GZl6(!ithPAC})pAI<S767n9DZY5J->&evKnR+`=a5Oz$DaM_<u_c8+HYQt5bv4TG2Z<;RIcwHEc23frS}-CVYy8^"
    "&ifD{lL2HM=c#Ki24h*^mGw<oUzOd|NG?zzjNn?n(N>trMA%6ti&A8DZZ?nF$x$mvFU+9CWXda4RywMKr^DwCXzQ#&TXN1K*}3OL"
    "9c&-jECO;(j8P2gpVIOZ$un@^56~wcefbx{0pA_w1L`^YIW`Sw%Y)@1N@UjYQok!SE7Z9Zot)V8tQHkHYY+5gAYURny+VM8UX7M("
    "-m?3qnvI(Zsf1UNN}{-qhe*JdT+MJw@B!gd2<HeX{W}mei=F&ykFOCf)<&e=Tz_hZIJ)=Qy4UR_ul25Y-*YtW7~J!cw7kJlKM3A_"
    "_ZVorHN*L9i|(0PlT<Fqp^B+$-Ts7)v5jg@*jQY1!p7o#PS^@=pA$BM?s394U{xn<1gJS-V{!cnn-J8Tun$D;iw!+E*C_+`qoR$n"
    "Gx+pzdw=7H2w_NO0vpz@K4>AbCj38lJ5V9d`PfFG9sd~nzglm`JPYtEUv!>K(?t_Lbk=E+E;2a9mH1j@UeI9XD%TO#QX?1D@Jn?)"
    "aY#m&7^sTZHdx9ZkvGQrUB^ctbh_hiw<|3U2vIs)N#hcaH*CQ?A$;oGDB8h$M!?jQ(i*GrJ#u{|8Ahyu7NFgY^@GN!-j3G#L2KIi"
    "yQb2kpebCF87t*A^^Pj)UDgilDCnYUEu;6EwbdTA+G0OO>f<iu&7~o2ongpa<)SqOnTL_FiOBt+EYKQnTXNB<<Q%I8IeI+L*n92_"
    "VU2W!z6Tw-QBtzpygz~|Y@(C)OFhZt0(gHk+owM<s@Y{#K{maD#>#B8t5=S3i1s;)K$sxorJm8qQ1HMtK&if)#XXYtW?oN`iPj5x"
    "U%I-s0_dxd3*lf~pehqsIHF)wQu>VAQ3koZ6~9(=QLeI|Y82SfiZLjGci+j7Z2yzZ1$H$Y#*NZ?`EryL?GFFp8QE2t{lL31U0_eg"
    ")}kpr_t5ipRb0CqOxS^Frz39{juu$NehFwMijDcG<rI?7Ze^*hs^ULX)r}QnWk=VDmMZZVEu(bE(N?SRn&KE@Q|Mz>vwMaNhOO(X"
    "1@mB<N{oYZSVWang73qga?2H!qD5GVui&x7t0D^ce3ucG$Ll!|oFl$IE#Skm<)&Dzuki((KG3ZT)GTHVIzc9{Uq1Qjo3Fn4GI{m-"
    "$;;Qt={q`Ypab%V&!r9f2$~x;CIMz}VXRfx&z9ND9<kFkL2gd$qCo%k+!R!cb_)pCjm;<O`HTD>jP2B9%+gMqY0S)Syx+Bxcnm;4"
    "_OCL+^M%nN{tr)r{L3Dt_<TZUG-Ym=)^-ZFh_CfsYkNCM|8Q;Uk%Y{XdC3a`hvO`cCJi#QLfg8`a;CO|w}EcZ)3V$77=6D@E78gz"
    "=J~~JQzECEj%8=Nf$n!E^L$OSKmuzVIVgfmXa$pp509IX>HZR>PdMLnj;xySa}^ZvPi4l>qceUUn{oGbop`^?NU9$9Dtscy&nWuJ"
    "H~AYs){PUuTcBhUT;&U#V0k-_BEufX-%Re;?S4(S!)9;Waj`tG!%aWP6n0RoG^aRNb?06&c9R3J3i7_q^7D%gKh*H%xT?B4OoJds"
    "^avx$A>Kg(L~K5R2p||=1}HW0RH=p|y%X}T?Yx{Yph7Pk#dgl1f~zi?I<f1>sF01I^pwncoag)D^i)%Wldl{c8LdLo)OTzLmUD~6"
    ">LwSfEoeUK`XLLLU1@OIZP`q3k%$>f{;4r~P*9Xhyad}(WuQDa%Ls2`OiH(3uFo<ygtTMrIeJ?r?o&1xMZ(ygPW2(m%Pp^QqRM?+"
    "jBJgF<I`HRCJ_BS){&sSkv!0Wsw2=X0geqG1kDjA>_-y>-IFMUbswq(IIGCwoF=O2B8RgNVN6H8(C2!g*kfm%&<Pf(G#I)|%bmQ|"
    "4jfF4I0lO1w4**B_bL!jVHF1dv>Cx6XuT6l)j$mWcVeg<iscMHRz8RXcSZfoJF`XtVESiT2SGKsKC~L<sKKnx(dZ=EnP%CVyd_!n"
    "#i5|TXR~4`Vio+%KV?@8##uNnj+zR4j>j<!xC2rfd2~cDns6te!@Ii|oOr?3@pK<_WY%~_sNNCA2fdT-qv=iZD}Gz?&Tjc9pFPIU"
    "jR*ofO44qKP7#B56<hv?ygh$tx95}2qgfSByye@8^!%1cv7<<MoXEjDkC|0HYS;r3T!K&y&#6$~uXbQjO~+!=T&8WATg*71<p$3l"
    "^EXS+BJ&0O^Xli%KCd>)=&C~;2o4*0g4sAZNrIst!Jr})i59k|t|-Y-QB&DF_5V(=4}#)Utf?)D%mw->1U;n~PgdLI#=H`8Z8}ue"
    "RBtY9^fOqed(_ge$5e0^I1TMlNxc8y5u51|uisy77jT}qj!p^nRhh)H5@;$$>C;ke#THf$&IcxiK_a$8G87H{bF$?n2EUcobKL*n"
    "+eQZ5rJV!=>7vFVd~IfEEQS9{Kc0|y02BEt>ZlD1;M6bAc1`LH#=_OV{YzgBVE>3*@KOQ#`N_Y7TUL!q{3S<cP5(G5`<H-mmzQy9"
    "azD5H3U*n!HN*}bYeSp$kqO+7@fHGj`_<lA+}Dk%*GZlkpY?k?G6g#DB-1WfzPb4E(Vd=-{%-8QweIt^4;}Y#h>ttKOBbN1#W}5Z"
    ">aOwTzN2gQAB*WwJF*JPS-taf%Y9)ycI)ozLhU*_+xWD%^P2Xs)Ow4%o|C={xbt`DVOi~oy^jmF?|c1wcaDy@byql#F1GVcu5Xkb"
    "No4N)2glv_A(!syObyjl7}j^)lZ5QAJ@s#2{riaiz40ndNDZP&T!kwz_)Ro!jD0_|)3JKuT`4dH>#y_ocsg|JXG7Hz(>0FW+Sl9l"
    "m2M}OL?$eDm)Ix_1f`Fw^%3u#=|y*KX#TPW3g%Nm^&<Y_(Wv4_GDNk8w(-ENPr1Nvy~=|QJ`A7_{@^^u-<$Sx)=9o7RzGG-BXJJ-"
    "m_Mh~r>t7u@1ZnDvfx%eSB<<tw6V13uQ2pJhQD^&*MTY$R&X$>Cq6ygTx59xW!vJA8^Ivj^!75$eNOpHWuD~yQ-qUTugwA64{Ho>"
    "S%W6Kw#rc}OwY=rq8am~nais3)sH)-rmEZyV21*BSPP4NH^62fA+*_~uF#cvGoBGoSg|+DQs}qcsKhG=nAB%niyx2@kLDPgJBMF$"
    "j1fP>lfTAgU+HeM!?*czit&P$pqrSdxw<jpsBx<PJu|WTovf!OndvQG2kxPR*1YzK>-*O~p{vKP_m%!5MPDU%fIdi`7TXC0r^n~)"
    "qNfjSM<0g$Oj4N0HL8VE4<D!vE1G{MG6(<9iUq!Vh0Qx#ZVuCVevWVNbsl)woG^viI-7PP*?=OdfA!1E>erb&A4;0FE)y1qAT2*x"
    "BKvNfQeq+ZCK*^7x?vMXsGIAc#ZwG@(&a#P6cUM&AIQ-Dy%9&KCr+Fq7lu<Zio9a>5>3S3Ig1PQx`DEq<c}yVbOd!1q&T>gkDjfx"
    "otko0Nz`pkH7_Bs76tbDSYq<?Vfbf0kR9f}7dU(;_?ZMB)`F`;tpdLeV0S#fjcYA<rm?AAtr+B7T;VHG3#$cX400Gio<$+mUSUD?"
    "QBZ>Fje-qZL6VvcCb`&@v1|znM(0uC{3)pI={ofpa?~CEa>}l_YJ;L~=uJr)+sK}%zKD=7jAfcqUiE!9JAwC9Id?0?9ZiY%L3U~H"
    "ZfR8a>KHi>>zf3Awud<-24c{sz+y5nTr4>)LxT<+sfn7gs<>UA<*?~Z_4(65gu2FJJx0ma%MPV_gqb9#pdZ1(3s3>#ow^Y&ie}UF"
    "0VJXwTnml_roZyFR6D9)F%tS?R0}+mz{6T#wQ3YJ{S^nlIS4t>p@uC9n9G6F-$TA}(dh#FcKMCkBFQS3mzyTbakb*N(^Q3EW`jR^"
    "oZtyf6z6d2xr9v=W3Mz@d`i>mmsO?It4(UKSe`3VP>?gz12OI48&I)q<%MNsM_j#3fC;2yP<9-|)+H7{0!3oypkM*0eYq7=zvyip"
    "(7tfr*?BjpfY$ZEa`zQ}fyW)47j7Xti=jM7uLYR=+>SXfqEB9ds&|#^wsWH#c$u3L;{+hx$xW&6HLg`??xg45ZF?U$dRsE_&gtdM"
    ")<llFoboN4%1*hCO<^mDh8{RiIW$w-mKGarM<(*OFn9ffc>Q6d{&4U5EL%JT{IOM|!r>5Oo`NnheSbeY2_l+`;ltWvlk^$#=0c0E"
    "$V})On!mKc<}`q^Z?BB{VVkn^WZP1+)V>3zVSs6PCro<o1S&OiAuf+F%J0N+IWZq9JWeDbt(y+t|HW(Q#VZ$UrO!#+-Gtet!rz1o"
    "qdp75v7wUdc5|G1U+6A<y#+@2jq`bu?b4e#fk~`-iz>G0Im72Z*oGmtKm<s`YAre=5(K`OL{O1AYVU!9sg5=khjN+@VWJP+#VuKQ"
    "64J9{%(D7}f41>~h5?YU+sZH{9+<mq+{<&jcKiy#J$P_uu4AIO#5OkM$pRD!QpLRUm7XU52*B*Xc~HwHoCUUXbKK%E9O~OC`BjX>"
    "OI3RA&CmD^jPyhpc~3%N{eSwgRx%f3ttQnS#DprZ8bCJ?icGEkmhKUw;FKm_*IrD|@7gB0ncS&S&%dqV8Q_(wC>_JtA?(S+;X|6x"
    "!8=2$(+@SzemEkRuZLdQu(C`us4DatB#$c5qNP-%Nw08NUFbF15M&A0C-&CkNo|ub<FJd^e_$Eb90oOqd)3r73sKcxjMdlGOWUpP"
    "$$@=vXd&QZQBKP{Mi5GdU@!rU#7T1)p9(R(@wup|I}Ga%_p7U>!>Jsdk63LnAre8&j8tOLQxf0_=Oj9ga8kk{Rn5wKxclpOl9$;U"
    "LwV|~wA$P3J>-;}&$G#fGJ0G5z5@b`C!LprN8N6ly!DioN2kF728Gi!-F@&6vJ|Rs^oR+Q=f-bo=zW+ju5be2=7oayJ8`WFg1rlO"
    "0rApc|2Ma{&|d5BJOY=eJ>8!6bbY#``_nELD52oR?o3X*BhPAX@>7c{$ST^(P8nOxF@bqcm%*QyCV{dqDm31sX9t?HHCDnKcGmE4"
    "7@ckBbMMAO-2t$jI&h(Jo$YBAxpQuof^LOqBOg#!lgKE}q~W;18y4hwyTlmK%Lz2Q#OJqW&8k$(L3m0J-Yo41yz+7KYiJ_J>JX{I"
    "IznzZb#6?Me~)fPr6V#?0y7V#p{hZ7;3bAf1V+n_O2b%;zL+4y0P>6>@h~KyqArr`DkYBz6Mo!;t7ZvjZoGdAiTWPnRPxd#-y?Lz"
    "8>tn%Yxzp=(@nm>hY8ubl&&5su{kZNnpJNypNmi@A61}iRf-7qom$~R2lKSr>P7;H4C{5$gAvgxxFsqTV;0Dz>8YF^chWK@l7Qzj"
    "b4`axC7Io{k^6`rIlod0RdmfAAs;|_G0-vdPfZ}j328=rJBcI&lEL5|{MYkz0>}RxBu2>@)k*!akp@AbSdZTf9xGY&bGepuE<&#E"
    "_<R0b;E<aZ**hL7=O?=Vaq5Y7!zwmirPbmvBPjH^B@`V@Qh@t(Zv7>=gSd#+<XMQ00iu}uak_TA4*8qzBIY0ly!tl6j9C46sNwN8"
    "1D5JY<sCqp=a>2B+JLg#pNfae_ekrp$%2#?=jE!vn0cmV>-3POK;hJv-Gr<yn`moP_dgjt>LjmBmeco_7?YkFfmDSvRyt1$HqN41"
    "YP_1!A6glIlU<w3L#G^76)e<h6f3mCEA$Pr20~bVP)VLo+0-^YeKQiCE)I6f5ouA#lu?Mn=g7xLP6!e}>xd)cA8L$zfJ$dO^=@~8"
    "B^)SZkpKv{(o8)pD#dlLU<&AlTHZ5s&EAZe;@9J|`D*f}xyd(kU{3uJhulsmF+?^UBh6&6UH?VjJF0!-yc1K?BvGL7a{u;Y^wOlt"
    "J~$IvF^9T0@7~<tGc-{f=zjVkk`)a~N3)yc|NTEA06<Yy-Y^+`(;Pb~5{{A8uX#ldc1tQ4^t)X=oKl!ton7ima>i8^TU<-1>nbSR"
    "!mg6T<oZBGh*h=D%H~uvi&8B2N~)q?1$N^nUL2G3d(F?6*dpNWOfFV%nrxopg)2qt0M(Dqq$pvhVw=Z%6x++@6jUU_(=z>)ff^9_"
    "X9FCWs}^uat++*@-Bcg(rSQW<b=SVXFOgVNb`F`YX6V+VQbJELVQ|k)7(7{BfVMII(OgYwdLrr)1CsW#!=9fW(D&`0=Cheun$vLk"
    "Hr8WWmj-6-Smjk8D4PKPd|$SkqfYXit@C<oe;=>cv{rDIukc0&q_n;R0Ns6l?Og<Ld~9=8!snSI5y06s&da#So(Vk~m(#$wFCs_g"
    "2{WCh=%%Uua7E!b+-H=|8OwBRUxXaVB8rjSn+W5$O`UZ1=dpZfB^56{9ChMpC-wuKB-(BYXRDOyrugCIUTvA7n#RPJ9j%GX<~oab"
    "JTbaJsT5YX89voiu066%Q>Vq4%o_9P8d;UpR5b#hT2&ZylJ>E)kBjl2He8TVfP&L%RaUpfVO15ky}Gglh_*beZ+TeTa(`FL{p#w<"
    "qF0ANf{ZF|C2^u_<WE%081aB_w`xofBo?ZK45*n#ztq75386Z<nnsMZalF}Kt*GJjGGFAN4l{@9vD=SL)lkd3;wjf@QF@isr^+hi"
    "6G3^)GEohR)~f>VK18{P!=D?P>6*!@5cO8t?wB8zvU^RC4*nO+km{qJqVqirO3bAh#U+tQmBjK5@-iNqs&t2ldsNj8yLS*q$n)^u"
    "Ha=d^NWYRPSw8OIN=@8g`PcDfb*0;IW(<o@Zr58wOp$i@7Inj*c&2tZ1#W$RSu7}+c6O0o=BvVYps0OxTPM_ZEvOHY;c9ci7ds`M"
    "R+6*pO?C+S{~`WJa!e(P>YU7ENd2HEG@v$feDPF;3i=?K6d7tX`c;*A@r1ALmf2-i!1<{Fl=+(8VJ#CtHhHsN<&^1C>7yMZCl&8j"
    "<fw6Fu?Uhe@<Wet0Ek*-p5Um$X1Z1xyliLJWCXMNIB2^&rg7D56ZE`kte!WG>3I__d+BqeNi7rowESD>J@6PZS0MTz`I-|)Qqq^K"
    "#2y2K0KZ@c$SYI=ffDQaYC}Jkt4(&cS^<MiS7Z%11MYg$afcVYrQJ0cB?Y`%Rb6)*qr3HPq$%M?fO||kQmwb6S}$I$zoS||UhUD2"
    "YLDE(vFhiJPLQN}c(!71SH)gM#s02}{fdf@c2#r-3EYPZjg-(bS)tFO&Sz@mlnvJ1MxNZjR*0skbykpR4EA)H=5&b<1yc&FbVz7H"
    "o-e0#8_jumL1G@RW`|SY>>@wglC#7)C5oNrKc-gJnYp8YM*LD=wCT(xg4FW%4jcxip3~!=KZ?c+QfZ`TpqQGlrktLz>U%J29Cwmm"
    "EVD!MAuP5_An?6y8h~bud3Q~AFY==#_q>xJVOTqXt?g`8W+w><EZk-2jiz=$P{P|3%3E4Yu&Pq4`iaEQw_2g~ESinKSvHafNn_AR"
    "K1x36wgQ9a%WuC)UOjvEIog&HBYNpQ{lzz5eDx)MF(K6&eoc$fyTK1Ltx<Mloe;{vPgaF@zlLxkrH^aqLgJ0FD;#5sdKCa*?39v>"
    "{@9XKz)S!pbbi{bCM&!&^Ni^*Ff!_DCuQASqk2OlxJLN!YRnJ9Q9n!3sd$`WFv4;?y#}11umM(;7kb~C`VsP2Jv(kNk2H1E=`+wD"
    "LDy3+e|_6N=>h;YrOeP)l^PpWs-j0ckK+<^)QydnH%2XQj7I^b<w5Q6t#L!h7dTk0R@1R7(kj~Yyx3+%mi?HObXZ5%S=2G&?Rgb}"
    "$FlqqYFe_E`!&ft%mF)ra5#35%s(K-f|hS81ER&flVfqqq{FtU<wZ^#MI)G1*AyYtMs=~$!uCSv!Z2c5q^*dbsM;*YHl&@Yr$3FH"
    "BbG^k!MB~k8l~u)p6hoc#NJ3nD5Ut@VWw-!x&a%y4wfiWPsyN$VtQ|+*y?Q87!|s?*O{!<s;Rl3TYMdcU6%B^x|Qayx=Ph{8SOn`"
    "rOPxg$RV-Z(h4d_TDj#dzYd)D@Bz=N?BK5sF1n^$rJc+gc4y*Ok-T?x);jVfdnuObrEskG#C`R4v&W8dhvk0f<#&%IcrP(?Pf_C;"
    "1(ArdCK(=_XI5TSI!nguvT35Oa<;t8i`5bpFIb-IP?Bg59W0fo>5{3V3Oe*4Eb_KZ+0|oJbxCdAVP)N6W!)Q}w6~M4qPn-*+RhUR"
    "T=SrPyCCEb<b5-$r>DF6@8B7wF-#l;Mfcb0ACsdwi1PQ5Dwv~3<CSwh1G$P3+PI(E@~kd}^~dD1q#sec*{k7|K;yVJ2-sH~J?2Dt"
    "41h$MWkrS+itKN<@Tp8F${uWU7+xepFSm@$U?5>S0v}V=TAujj-Dz>U%e*t9ZL#3UY&0ri-?mtGS$AS+d^4EtO~u)>wkHBb-TAe)"
    "8*Jef9dQi9waq(KL!)wKv%pHEBH(3ag-?hkbX(6`)Imi1Ddgvx6t(y$W{#nK6l33FG^4A<$xv-U$|&AP7O#H$R)yAZui>!hYn%vm"
    "+fLQWXH~T|oxZzSqmSA$U~JvGZoc4!E5_D}#=CwN2_OaMZFB+|ywhyL-bUmZqFjpF$yydGmk=yG#=_~3vQ#^QSa&4ZgRiXx>WMnO"
    "IvpT$Pjv^lxT*tOTHgV3C`uHwHSrDL;p=D5TTUmqw5A&xpsFLfyuK^(SCNU+74@pHvL~v%sw4VQV$hERvDJ`pM=O%@D$_d)x<PN|"
    "*`V>oFTQ<w`1I>1uU;iDp1gSWGI{>wt8bE5-#&l-<mF#C9Dh1(@A;s?$*kpw37ZqH-9oYZKF6bBub)AMJLPGWKE3k_VUK+%rULH5"
    "xpP;FsHBya&mFsyhpOI+x@()NRF%ywWps;*^&zYm*`Ffq#aviaqhV3B<q8ZHWl|q?f~!dJe4wI7XcxoK(bSp@p5wmyk~K?Vhm)La"
    "zdEj{r?=wX_HJg}pUv9!oU@BB+ObLBYjWF_lV+=r++B}em+KS`cM%EkH@y`bHb<%02g$Gw$#5?u>XmP;?hq45l_RPw-4o5Q7R_)!"
    "G<H-q%G7O^;U23{?W8*9)FE&39AP%02vl-$I!v|y>WmR~PgdEN8TOs+eW!Ep`4Y>|(K_uF=G_;edt<9*v-%K~>E@Q!0q)fR_kG}g"
    "4e%o$_|bc3NQy1-BGXZupM*CxA!VyL(V<t{B~jZYLfajtD@xft@m{m&`-^HSdQPHI$_qbHLwxHT8mr>^hw4g>Q)op?!$?cRT`l#a"
    "E%j@wxqhIwqT~GeQ}EVbq5oTQcAdN+zrD8i%JNpiEi#4mGq`KA>Stxw1nj1Cg|eV-2GHZ|u3R2*vMWx*uRb?K{p+A&<a=Cwmad3K"
    "dA1+pm8=`0siHsqBs}*#LtQT~7rq#iG~bWq0*dY?=nlHoQm9h9thBU7jV{lw=&mn$x`iWi3YshY)IKE1bMhfcnjm`FG#WF(V3ea1"
    "=ZkqlrCL5c?2S(2jxD2>g9_(lA6#)?mywAGjF(hnRl^?r=_{Jq_3=Kv#;9}XG0|9V7vo|}SzM$c8g?J`ldb~j*NqKox>^IQK?}fd"
    "8~6}|PcVbVb~8Ktq|r*!GMQO(_7CN1*_mz^>!#{JJDIihjb2(#@>~bb&B-XOEsFMa-z#WwYWzbsD|}w3JHeKadz=41?j*m=%dPP%"
    "cyA9%#m%LdidETJq;E3#S~i)K<dHaDy%FJu)eQ$<l(4a9lA_M{1Dv^<-%(<s)ndI`0?|sK8l(PBJ$x&fkG`vV*7qHn<iAS}lkZeI"
    "5PFbpq8FE<Bj2`@SMB7N>Pg)keMrtb<>j>b-2~=xJD)R0VqjbGUP*Ny+WAYUdwL>!JL<gFLz=HPCK@<wF~wz8Hg(pzfQ3~eMDc9V"
    "&&F*72G_ufM7r8WfPPi!6~zNy&o`ZNdxkwLo4t1OsEx(s0^RH#x09pJvA(Fr8u;E9VL6E?MPR7Nj;&%@3`B}gDroJ2D@c-UkbQ{<"
    "rAln*@<moV5M0{vumbFg5H>l3X{7vGm46*ppJ(S7H$hURx0@<#S__GS-&=cN@-v#pND|_TQN;LRg`kyG>A6&S5=RlCOka)1GHJUq"
    "S+YYNlRH$lQH#w_SZvU_xk&fM=ca;&4dC#w5p+?wzucmSQIw{e5u9Qke`#`fCoDlCpq2C3cjxgH1&e){#OAZ_&nL(Q5EX0H4H>EI"
    "(slH;eZndmic$^w@@l(j+RrHM;9t4JPS^RM-^X_tXRE7mzNB{<4XUs829Q=0SkYM%s<xz5^%MpW&*alPS*Ul8_`?^lb#kyT<zN@("
    "U|+n5Pp{gT#Evf<P(}SQUpEm`J2~~5#fUw~EqK_)83rkA)ddQLlU{XT3%XdnZDO^wT*HnxXr$ZCO59ZgkmUh!Ga<_Kq`v?yHm7hb"
    "Y%B=?_3Y!O7r=P&JSH~sl^44V0e^@9_C}GQ(9rTGyV^8=0L7<GK17qnDg63j1ZvntI%rJDZ?l1+lEojh73`ll2cL)z0`p8(^Hp({"
    "7EMgUlwJ*RK4zFZ0T=+EQXM^RgDNqlpRL`@YnUg|%=0_vU$!&T83HvFt=UEt+Y5T&F-YuNS7a@=vnA>cs`@fCU;6xuPo>JxqT_G~"
    "e+YJFskf0hP*IbJ15siB@UaZ{z|!9Z6ES0q<VyF~xmn?#U@|W)`#Q*2OlFGQU77+Rg-KY`tG=Y%6xs3|P7G()JUb1if?jRUF)N2#"
    "!v!?!SK#>FF(BQll5ZfpXTnjm`gCOza=D^SMK$Bc4_Iu%x!=jD24sUoa0h>ymJ0mC+5S|IFk;1hI;890y+1@E?VjV)Xn!{yk3P3_"
    "@)5KVU6I7OZBKOFyLE(!Vx5M3vg~pmiV<UH9ZLeIVM&Dj+I5D$N3(9^8WTE8r37v%GsKj9d%K<fql3NV*r)oyCL|e#vms1vvohQI"
    "!*x~^1GP9nqe<6<=E$g|F$bK6i@j}`IYi4}L%hPRMxI$ze;+vg)oxQhZv8#P?)uEkS8p?{fjMJZp2XeclQ(7aD*VXv(3BaPUHShU"
    "#U;OsQkF!E+NjOHFA44?jII>pDm=f#g-0zBOd+LkL~<$4>`TaZ`Aw9$tqQ5|+wcr9Z>77$?vBqqHx0XJFO}|sGuFvFVe}U9qvO4G"
    "5q}NqiBc>1VZUuB$Xywl%CU8UTXhue1B=#D)x~Ql6GKd!e{5x`1}fJVQ1&U^I3iU;3P$y);H)Z_rM<%ptQ^0uWy#JXXwAgeSomU9"
    "S?*N`g23!o6UAxYkVutz7`(6@_l&$$oqD079TAV|gXGorj2|O$GGV?LIbbhSJPu=I%Y51FAETQrZjN45rWSwo<0t7`2c&&nkhXre"
    "|G4`}U&*_CdPRX{bTu7+E#5BiKlc&`O;aq1h}h06um%0CM<)|;{D3HhS={T`DGg#|avAfPGOXtJfa;ywdeqvgRAF{RAS%^>Ku%D8"
    "yefsyX8%j4Kcsl3x7;6zY$~h<Myik8jAGW>5UWKsRhCie7>s)4=TyQOSWy1SZfK>Tc15IJj)5w}t41yu-1l-A^KJ*?g}j4kwSrO8"
    "z5~N+Dlsgh$C4<U{@`3r;9ER~J?))fqD20UG0s(G5ypO&7FCrEO&>n&OFZWLp@fLwR+k{qKNY~Mjm@dw;+e(t2j0H8FClZ>GWwce"
    "c0x_Rs&L;r+S;A(Oz2=S>5_p@-78Ge%_b|FdNpJG+pZd({u}3PGEgfJB!jkm#jZG>Jx_Fu{^Z5eCpEVmFP`>l?l@jN?F(~*v8_Sp"
    "m2r)#)LGcnV0(J7Jv&$<g!E{g4?J8gOZ@S=7Z_x6sTy#vcEG(o2i&s*?(H71Hj`srGK3rgzo9V%e*yz)yNLcwzw!G>nTh2}Xh%4B"
    "RCx*D<q6U3U1RDe^OJy!SVdppzT<_oH*pnz=LRVZa_9s~3^gV{81{cAsuOXHiZ`udFY*td#(R!vN&gk2)6MFLjp|oT!*>pEshX~r"
    "zHnmcN$CrJ9~<eLFQoKUcPy58r;22w=%q`Az{Qc$_zvg!n=IKb^G#VJT3=-<GJX}H5m-vD#pDu<qgoL6jabt0<MM@+!)n-K_D=Pb"
    ")eJl$d3TMxx9`Y%w;6ejL=H#(I}uJClLOftRKfyidQQ2yDZ8Apv=q(-)7Xfv;Jdiwp{M#D0Ka4G8Tn@XYjh3mezFdO2RI=CYzwnS"
    "q>b^r$j6&T%N*8TX4^8OkWT3wgQi@YsA^NM&lupRlf2Ne#unRh!x8EfCeRYuS%JLH5<L1*Y&-gZVV>}Q5dH-bxZ!8Jn`DKrrt!hD"
    "b&cwHadhGoKd@&(-|r^!_}h)5poj_^H^$Oo*9)%OS>px0kjC{9@G7^Idq&o)-b3mT*Ik#}P*QC_wp1-~LUay^s%|)=YoVqYfr2j5"
    "!bWx(TWXNos=As+Z&|2PCE1BrC7TF3Tzl<Sv4RP$S5X<EgD&qj3h&p5TiY3j^_HwcWzncsJLo6bXx-YW@?FGZa}N*8sEg&L;{vI~"
    "xNx|Ht}B1b`@37xv%Ad1HoLc@B@vVbx5e5#&~LFhaYS2ZhZTRB?QhVSt+oZvDsN=OMq0Si{BRW&DQ#fGIPLY!mO;&Csp!k8;+({_"
    "_}_fS|M<Bn{BUq;L&ro0I83a7z$ea2^m1_8Lk}L%4<5nM{OAaDLO7K_h6CKkAGecFK4~YPehPYHucu5@>3psk(;A)P`w5g|gK-EI"
    "qKyC!Dq_lcZ;WS+SFfMEe4RZ1_UqSQ9a3PqXWs!Wzk2@co7c6W<xG93^GZjfE8SAjq;XxMO&7S4z%Aj>6IKk;pErs#%bA2}#ySyo"
    "!P<c^!rykYrd%*ixIf1*W`o_inJ75$o%>NUq59V?bOWjk<WMZ2lLkYAoOrk<?drL=Bl9Z7IRSngR2v4NG>%g6qfOE^aUKB#Z{S58"
    "{SIDui8(&=4ye6-FsZJ*sXvi~tDsD<4rzQT$o%A&_|X4ZQLKtE!(~(z9lWdHuN&4S&(Ps+GT%-GA17}wvSnP@G`z%=<|Rm6jo;0R"
    "igW`>#(HjGwyZc3>}I4vTv2k@lG?GbyB)Lh*5M{n?^?-@bGsPu{Wrx=I>|4OfJN5%^e~<L?KW3tJsocH{>iOOuq#m8FiRdMIkYJp"
    "zgl4Mk+XT`gvwr~Yd9}&;G~molIber9SP+HoKL=d@vSm)u+nf2{l75T9D^v)Z?bF34z<nv2={QPTxLzz=@OvZ42^6}opmC1QJnKq"
    "RnlS3iGX7f2#|yIQ$AmrhmnKQVfu59*?F-l@*m+C-8=4DiqOAbuO@HkoHxr0SZx5HDx+>>Q(D?46ZC-M1&F1qHIgQs6SCuO_mFFY"
    "_Orn_Ow&V9SXQX4q1WZ6Sj|y*WJ{cgX@^DJYYYwyOo`4XB_YkyVt$>>;UDW`%(*tn%Y)9DS&;$ESvE=Wq?)py+G=^86^RZ8nw)K?"
    "=Wv{So0b$S3QkGbs2u(T1@3gLI?XqbxRvsU7W=MxR5b6LR_CyiO3Iw%pgH6WN{jRguM1)r2!y%Fxv<24tkPpz_zc|%9*GMx42&~K"
    "RD4Df)C6;M%z9K)p8;*rr(G_R0UWcr-^k;~yBq0OH}Y|`5xEgoA?;No)q|2q)!IAT=JP46fjJ)Ert!rvkn6%m!5mQV%QkgLEiT4("
    "dBi7a#zSTvdM;;{%@>#v6Q$Xu@yti_YPqpViQ1$;aljl3cW86Ka`Z#=wgobfV>C)|6B2O@8V`9UlO8wjYV}bBO?0CI<MRSVG8?QB"
    ")KAgQ13(3sdNSg#d6lrEQr``VUT$_b)(;w^dOKR{2d$Z#i0Vp@f~G`(+M0St74<G_2X+*6QMCrc+g0t6lVKF8&lw+_xiq9@PMqae"
    "nyJ|U<la0h_Ute+iZkqeP)u*5w+b#gm7HU*&7I__JY(;<GlVtL6@DY*cjQLd+_Zh`eGp7x6P>hQ>PaRS!26@wKK+SN%`U47viUjP"
    "cJ<2iNK_QPa=+G7$}HZIh@Q#IB{GkCz21s!G8VPdPu+~^r2F<d-eD;?;!}HhdrmV_<-K+AvXb+Z6uJuu1OB@Y1+g0y{HS(qBmLT9"
    "VFwxg1hY5|sm(fHqS1Df(-XTR9Z+A7%t_|+`q}fZL7BERSe=!aL!6F7>bD)C%`ras^s)VH?kflG?WH}%(?u24LkRn)jHakO9d+=;"
    ";LzFf6I|h&0~i9{$o*0FUrjls@xig^NB!!y(7P|!v3fC$R@?JTP*b`9O*<zyKOsNZpu&Fmn2`W)p2gCb0pFbAfDOPU=jM{Z)zhB3"
    "Zy-Mwlt%PTz}*}~tRkSQ;fWECaNIKy;MX^Z%o^V=%k5fSY~gv>L3EP5b3feJJimA}Vf?2q%;><_xUO|ifE<1={#kY%G0Myu&vD%("
    "%6fN^;_)~6@Ezc~`A~<~ozijjL%(kyc9MpN?8#-C&*_Lw*@KgJC5;qkQzsd|X~c~Ul)*W>9x48S(<9#e;>z|oWR|5HvfR*$Zzn7)"
    "qO1Zcw+Xog#fh%^7j%S$V<G1}Q2Dgz4I-UB2{R7*ZCZd94;*i_ca5Trndio-`uD7Cizia(?PLC*Z%oqCTu0(W(pg`BIMBzUc?WOb"
    "nk{?D!cp3ntMPe}!m;qgu^*6h#@R*+xR$sp5T?rAl5Wo^qcT3-qyG;1_ZZHhY9K@X+XmlKm-MANGWGP2J`RzZc3R{-;53TR>~|%E"
    "LKyEmtu0lyUbLs=^KuYowV=b%n|!TPdqhB7k4+_FF!;_LgpliBx}4(OCpym3Ol`7d8MAwyq9vf)HpYc1o&U{4apP@DRsoS5J)DkF"
    "ICyqY<qpvMn-GGCJK%rnNW&YxKPvT5#O*A)r8s!KG!kl^XnVGP67RIo)&%r3Lb|wkrD;sax;w$oB>1ovoY$=ezmCF+1>2NsxmA%Z"
    "j*>hP?I!7_X=MiPtz9o^;u8l}OO+VpFn~OZLaGRd7L<qRf@*`}EGHp5r&xp>bFaxNHD=c#1Eu;!sv7?!qlgMq=$c0IO@cWuFqaoc"
    "H&m<gFO*PS12Zi$%-91Z6!e=@d>8?%(j%esqRy3sZJq*6N9<WlqcEXdL?jo6E;rLiO-fqM6}QVX%$YaU*TxQ_4p48g9^*poWry1F"
    "Q`wi7Q`iqc8*U=rRuIVFi`9ip&}^D6z}mKBL+haFZmUuNYW`p(94fOGcqoB~wZMu?6f}pujDoAwE|#JwIR!CnfelX+0n?vb_COy)"
    "roY=8%r|515(*^BcBWngFne#OX@|=>Ck`xGN=q+_;Y5*Y#iZF{TcX~W?Jlbfu+(6&JXdrm2eJoZ&biVeU2b*0F*fs>HR0SpR`$l-"
    "w*{EMJWl8HYV10GJjWQ-`pye)A*)rO12*_tI&I<WO}>CRK!Y^>?mU^M3wJZ4#yWNpqs#xCFQK?|^P-)+&B!GWHm~$r6v@w#TiOXa"
    "_!Ov450d4w^R!r%<ukl6SYKmI9&vd>IaNrLZ}V@;t4qHq0*ktmZzqSnP8a{{s7(xF!&sR9ar72%MtHz1`Q3fPkc(c8C=JcsSytyc"
    "r)ZB&h7Vh+km8~CU?|gMjHl=pFltK~M_*dueJhL?i#VB+4jG{XMfj0GLVW;x)xm=kD#DfHFE`oRD`c_>byokcUad8^|6~qysHeq8"
    "gm(NE0R!6%RAxr=I5dE>Rw&~bWgF9kHEmUp^=_o)XofBSffD!pXNHqQYUmQ=4+M?_Xx8U{A0)jv7_A%f-Cm=vldUP04$qgoee?uO"
    "2V>ibY(q_p%!l-5efNy}>qKt>k*VfObm#1EY!fgUC;r@e3&vo6Z|mhI(mkCzHH>K}vy*#;;j&kK*L$<<zWpXFGVVGysX-59$22SH"
    "kXdHM=z<8iiY2FRtL&$io2dwE=u7JD5_L+RPIXLota<L4s!E^4p_bXkH5HQ~@Z=Wu0nzX&RUTEJ>BwISVVD8f7~E#dZ(CVdTcJHO"
    "rb8Q?dK9M$9iE%aC$i?s4{SXRD59f+5_#GB3Zi$42a$F5<0=OoVLK6N;7a9LfWgRun}5tSLz@j7XG~8?$GE~re8dvhEtfD??c`dA"
    "zKXQu=sS!l`HXo+UwVt10QEAHy5ePs3an<0C*OYk)aH6hH|Q8pZj#O0m2l-SnVwQOa`;<m^Qs5>_f@x*e3Ue=yC5E~d-%KCId;a_"
    "yaI*c8qP>qqfJ0+#MD#u#*%;cPDZVRz>dYVqdq)LR4oj1+QkC^|JkL~&n<HRD{w_==lb#H0<V`>^J#oKO|gz?hW_*hP&kxSO_Sr^"
    "#l5AHR`N3~4E#2{a}Hfv8_dq=*%gw;rE-KGjk7CUsG=)Plkm2UE)vRVqO_+-id+ci$VtX-h}uVoFyXSuK#82EXqnWmD!9v_4T?{h"
    "CTpv`b&n<owr|K@VFK6bi*$4G#cEqTT`e(s*|X(kGh#od3%byLUTc)+6syzC1r?jW9DpadpB9@jDW)3E`Dnpalr0M?KX_n$AFS_#"
    ";X4P|LdfNmeu`TGOfXL}jf(A`qV_5;DOLm{(0_U9qO9Ymd34^^w;Ma|at@TfIp*seaX5hw@zcn>u(ebpq8fgwuXn#IDs$Cpd`W^U"
    "R(2|p(VmTi68lJZv;&n~I=E)Vk^!}m&oAs;q`%S$mt6hWTg=w?l8KUyZ@Y$rsM-tODZx$hp2MB!P6bk3F5R1!5$kvbAzN_d6UoBl"
    "pM(fbBFPtn-oYZ4A`PT?)5p_><G-S(wet$g9K$tE$U5J!-a!1t&D?H-uPShg|K)v+RZ#)f{W|vZfyf`d%%PWK*)JER<KIIloj9dl"
    "4Qei|Vm^CySdGyFmNYH(No5VDe8JX9e)l2W@m1%r<i9*hUcY?u)i+;#^JT)`@2A$geU$tvdH&?P@wYENfAad-tHC=ln>S8`a^y+n"
    "UuRD5%DYZ-03Lk=JCVUz`hB4bcHD_Fenq~IS7l@oMl2i-O|`+<cFY2X*EiaN`;*<_s3v7=0OMTTxED1H>wHm`bYWVOwj~taQBw(~"
    "N#Ahnk!Iu;%MNo8AMrp8{=mKV@Ch-9(I#@k6q!5@k4Hut>M*)HWK$Kc8hEbT`55#`YpPK|e26o#iOUdS%)X+I%gfGo16}G&=J}ea"
    "Myq-p+4XvzZYv<RvdcK%Y3Hk&1vDQZz#s55h@1@aFzuylAnIS7Ht5kH-{6l7cQYJ5BR*~=f8(2Da4B!g5G8M_&NW{bt&0_RU?wkw"
    "=%tf10w1Q{qO1Lyu7=4dZrNGd9aLwV{<SW4PO!RD9B2EBI=1I(*mV&JmdVTpVn8RzURBQ>v#Dt)5MdoN$~&yMh{6~wIfC)moJ}*V"
    "xxm<_c1j59a82b}I-3~eO0yX04M#b)hk!yE8rAJiRk7}M9d(A}fwv;{JT=Ozs7KwZZIPKVY?9^|A5P#sP;JvO43<R<3VB+#+K13W"
    "-6OJ5rA`@m6tz{-!|;>HlNI<L%ZQHsio#=op9bB%Q9@mDS^cD(PGRP-Bc6_9^o~?bP>6B9KSuUXt;WZyX-yXq;Ff)4oby}Y+}TUb"
    "L7u{1xvT=dmEP7N-(D_TJL+-|BQ@t$HT|<djM=N(b&m|ux_-;7_D5d*eaj2y9TVxKdo;ay=qmcZBFoL-ofSAIpFPIUjR-o9C-M5("
    "^M~jbIe4dJ%}M9c4Df0JLVs3ZP_w8^sf^)rDG{i@D(WLmRIRyFi~sW9AXd1EV7#IeK4~b4K#NB&cVbfa#Ri9XIFaTF8T;Wx8Y{ww"
    "6dYmb#J6HJgc+l8;utk74q@FuNo4=R{;Gf)Cz^X?JuMKimi~>GDOQ3)i8NG-y7It;_2u8L8;e=zmq(7N(FANTCYLr{PtYJ$8laQF"
    "9#t8@gJwn@h3QS2IWRNT?8{;2vuo3^9N=49tj#8<7qPDAOWQF?&e4Hz+i>l})pbs!$`ai6nL2hlN2j*_$WW@pXdoNml%vJ*9Z)i{"
    "D{$i4<JG%kE7~hYian5#gXSIZsSU(c-gb46x1qgzsMM3XF4ki7`$#N&3gngM{7_?QSX#p(v^x)fl3QjgmeAF!DZbTJ)wAKWh#z*g"
    "=hQM^cC{5;)}ADCj2IEeT&c!djnsEWT7lrkd2DY46Wx2b%Qlb*Cy~8()_A#Hl9w^N)uHW8V^sGiT8n*WQQah+PNxy^;E>N0?RphE"
    "!N_|zgGV&mYkI=ijp5bEcJiK?AJ4<zxnhnwXt;}2ZgFT3b<jE2t@`m}fV4_6zkS!dMSAPL_4n^v@bcl_z4I`}`uAx6{zYD0+`ESX"
    "&R8FhcJ=Ybn6k;NYj*F`D|tH-+e&dMU*RoD#4K#XQ}T5zBD7-os-<#kYT0j_a{Dnlu*aM#JW}}XKxA@%)OL!uk3G6nkID+^9k^_k"
    "+;{Gk2t@}Nr`tK+!Bl!f_~KFEZ;KgdC$-jl|8C_yv@ND>|HkdOiMsb5yqKdqQQ-c)@LA<{w1AC#oNmUj!Qk-$IjnP@ZJG`I&<5?h"
    "#ea&$6bxif6mlK;y<jTr*A0$d;CEtQ40hF2QR3J~aC~b@>_AaR2_14n-Hh5bAD5AM59|v&zgKre>BzmRUMeN=Hr-S?9IHBcON>Ga"
    "%Bxruh{pX!9|=Xg9P1t=^uKaDo8?#JNnZ%(osP;Z&}?+jdu=Q<JSAFHHWqe4vfUh?XXAH_^oD%|D}9vq7P6*F>|)gPhIN*os5_3_"
    "t?G}m($F-vU2V0kJL<YQ8h@hDr3szx0(*nWqN4iO&~AB=;Bh@KcpRwnFQ2^l>ht8u)BobzuU>ui`m0}jll&^dm~ex495d@z$@5=)"
    "{%r7$8o8;vU3WvrJ)^^cv{a-;CW5|d4=Z#4J-ie-NkuRv>vk21N~gzSZ6vpL5gW*herttM^Msd(WIVBoLzP+R!1dUzv=7;z(o@HT"
    "7xGR<;X;=KRyLCT3?aIS8q11jl)5$EO7qG*J5MLq(w60$EZvCf#(L?S%{Wf&iQm&2%Vc#>bl1!oHE_r`a5Iv{`c?Xkw!o@5ePTd3"
    "<sx=V%kj-2zaUyj^0VbZRnc)k5}lwBmS}G+X+qGIt<s_zKVG|`W++e^z<TBxY6rGEw7iZz*A&&mh-T$mF@iM+d#ExlnRN}=!q>Hp"
    "8A?)IVKf=k|2QD~^<hl83?t9w7_K#+z}{&>HLD;L@#=(rTY|ucUDDQp-?D@JsxNpMH&&0m$Mnn_`>-tv$`o?FOc(j2+`DQd0c2KS"
    ";EmO4-qd&UXGOZ4T<AB+KCO9MQRHAtO?=3D{~>~vW6_z4?pmio4z@OiCB#v-Q!>y~CwF=fN&}T-XuBno+|}xijI=<5Q8H6sCP<sU"
    "nKpZA5DCmJ-@O_yEqdQq2K-N#MV^Vuxi}bg_+=nnW4D5w#^e-6{{9@3Oop_PA;$zYZiG1LFwL6I*3(k4&!1-)uP!ea4@nbK3PcWc"
    "2bqF~Toqyu^aLF@?<`;8tv)P^1ZI+`-BH+Hi*S1~)E?|o7*r3qDL<nam+ArT`Nc+gMx13A>1DnuRG7XGlGSBa1cCX`=m@<jvKc-}"
    "!bkNKjjzo3xrMqn%2X(J&lNRp?>O;N_iAcshXy;^@gl>G2s38Pubb{3)N1WO!^!dPUSDOwt3%+q&W7Rl$zj_mZ~1jF?kYNv$*|cG"
    "$5$P<-+D=ol{%+l_migAS#KKBpPD>aFcfI>k8yjG_k=OcrS0e-d9hkzl<4sS0Ky#+_!>(BO{Jp>I`rr*_SiaJO_x;G8&=jEM(f?&"
    "O(Nd3eQs(U$h6pAAwM{Czj=YGjfXCUqE@lI9k*NWQh(NXMzJ~RO^&{TN0mzT;X~Iz_u=TIb2Pho*boQ|sAo`9uni|*nAf;Y2Kkr0"
    "Znv^|YuMS{d;q&=^T)fJKfXot$Fb(k+d8bE4xL*5kC)p82z&K*=S0FU*B~^OD4*41pi)FwmoI*F_9coijqff@*nS;l+Kn(BJ*a*V"
    "IjD+A)C$;zQj(TmY2k+WV>T{f4@e6>x}4S-e03sk`)yQfq&11V;}znDRi_bKFxqVlb(@P<4Gfs7Iq_{&Ypv9cR8$B}lQO|~RSQ9l"
    ";OnT)^yrP@xMmE;u`wKXtIc(60aPtGUw2szbt|z-H<7fy+6o89DZL8DXY<wMO|$Vb2wtrfw6f9GJgsvw@%u)FmfqHhCyvMvbXvkk"
    "&nbr3!j$O?6=s@BTvOI*jETeEw}VRlO4(IsNgWyVyIoAeN)NY=y1iIbeV`)-sJ+uDB-`#girdtQj-nqMg|`tt6-Q|`rbLX1+^b4w"
    "d#vz#bGFGQB?5%)qLutAndY<E;GN-s(+|ln0l%sppdEx&TRZS;wqYXlF+xv&bc!dd3t-FfkIsG)8?+z0P429*9^3iex6iUhwAT`#"
    "rr$A@<qL{dZq1Xz3<-zs$p!jASVOx1j>s{wD;~mkxmu}(UvZ<mbd<Wi+1<VO^|$h3FLQh=WPaB{sJ?+cEbMhHL|Nx&gLh{5C!Icd"
    "O=9l6MK;aT<=`DFB>Y#?(x7$H=>e?IIZcGe*W2(<Ud})_-ej^?>*nD*DO+!N!r^ceIBU*0lU=kQf>yWbh^$uK6Nh_kic1U<pkmtF"
    "tYkrA4(2e-;y2dj9Ajy=oGQ<0Fl9eWoQa9^w0AgnV)ME_il>!6iVkMqAT&~89;H{4GZ6b?qVS%R?%88TJ&JT$JrWfkTOtN5%wbv-"
    ">9vRngF8z&f?bXn0>7KBS}qvwWF&dCjOluZE>-w+yoWxijGKPGO_OQ!<4zwF9e>iYLeyN_WA;XMt3`%18+L*HnLjxC`1ttorz%sf"
    "ygF9?ij5EY&yI&*e4&%(Y7j3)INnv@Qs9Q2tlB!f0R?&s5@=s_a5Qc>1V#wV^(HF@jTJOKe>>>oC}}(cIjRBu*FiYT=kxJdwt1U@"
    "00<B#huBs1`vL!U(5CbCMLOtqdM5pGC%wAB2qH~Pu%MwEAAEZB<k9dG>|Hr%93FBvd)Q5Ni}>`yNd)Yq+I+2Jj_T#(K<Uh#KhNgV"
    "L%iA}%l4tNY$s&d#>)E9kz1Qng<9qNnHE*B{b72?F!4vDS{iVE2U4&TsbFsbsz6&NDByrvnn@|MhdeFov2XIpn=%vwX3+TpN!+Cu"
    "TzykE&(7(213vLuEoGM9J0uxBx?2Wh?e#VmB&gmXOcXgQ`&`d*KSqZU3VaRbNrozCakebEc6~?8-#AbIO|Sg$h$;*R8NZzyIMfu_"
    "<q-oWhR&WsfJply?d+mn=!%IIdq=}ho;(gKYA;4xHSlXRLMf>H^hqyKSDVcWH%6@f#qp=ly2FMO0sRhQmJPH?lkx?TSZHw`EF`5)"
    "5@e}u)W68~qPG}tHqHKm0hmp2DZ9OxIUNO;Cq;~nDmR(;CCE*OD#dvBE@Iq6yT1@Wx}y-khtTG5k1ERQAqRn^iK`XYaLTj7bu}|@"
    "vnf2bm%6+W6RU<qe+)uZIp`R(rbw8%udmL52@!1lb%ZG9Oq=mUbnSy%IQDQ{j9(tf#E^IN<WVUi2Q>*Itt*a6g6Sl4+~L(Eeph(Q"
    "R4lk4AEi*{i9wwQ=*HXWP<lI2hRm1GpS=FX%fB|sN9B2A_obw#c0e=FzI*-bo6nzpe)#OW7r%J*?aOD$t7p*A=Xe<c|Gs|ti?4%w"
    "fxRr|CVi>qm`Dm!Q#6R*`8hdS!}N}KO6};~Te{iBO)}xgc{P?A(aS09q$GqPVPfF`lO7#s7LI4_wEX0~b*WUABH`n2(Zc{=MJFq6"
    "Xx>AX#7{YO@V(Fg;l^X+pz#=!w>HL{>37_aNqB=qy9Y9krc4&hehrc+mq#LR21m^$;d$UYUP{R-B4c<lsRqp+2wvPSQ2>7o|Mt*o"
    "i}4EmTEQErK@;ml;N#Z20c=uOq1$;ZfsNCQ?(lK98!6qZ&+X(9{sIi50g;UEkuXEgi^1dUP^3~3iGl{bZsa@Tcr2>O+&u$HXD@}6"
    "Q^`;7G&rs*3+sld>SQvQNKU3fzw4LFaUKVi<%q(%72T3CoYC;_mjJDM*<s)R?8K_>Ig?>htTs4fEVCep2|i?$ViCBm8&wq^O8?Nd"
    "bSQoC)z`mxJ=U3>MK){R8o&7Jn<rn#iuXx&{Xz07+mLS=N9M;pI8DgaDM8n@B;8K)O^2L|sNaXupOAcu0C=k$r9Vh8*(SOysUtb8"
    "jYHbf@ns`<HN7>i!n19G?nEjmBm6g?<r7ja*@Niix8IC;=)d^p>%Ybr=BOYtMP7b|8@m~_q!0)=uO{u|f=cewEvx0k6+)vF4xCmi"
    "G8}OfBC8BV^ozEmOa<y>s8Z79XFJpgcFEzf9b%Bi-N||E>ZOA7kUNxjV71|<lYF%y2eZ7Sw5`e6HKs&JF%+}Pzp9Ys8=cb~Swy(9"
    "sg7Egw&brcfvm~{+s2AOtCCn$;X<*UX>A?)1`Nx|VQ1U<Ts^<vtWeT&woUd|gT8}qlS3KP-Ex}N@y1YNtdu&3=M-PX;7-EbU>~j$"
    "oMriUkL>3@fB74fm+09tS;Qq&ahgGaqtXVsNKWa!*e>ugt>C{MJji_b{{#7<9K-"
)

# Exact locked-V4 source names are deliberately ranked above generic names.
# The UUID is the original user-uploaded V4 source; HDGM_v4_rapid_acquisition.py
# is its line-ending-normalized local copy.
LOCKED_V4_MAIN_NAMES = (
    "38987a4e-6064-487c-87c1-51ea27751072.py",
    "HDGM_v4_rapid_acquisition.py",
    "主实验V4.py",
    "主实验_v4.py",
    "主实验.py",
)


def looks_like_locked_v4_source(path):
    """Identify the full locked V4 source without importing arbitrary files."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    required_markers = (
        "class ExperimentConfig",
        "class SharedPathFourContextEnv",
        "class HDGM",
        "def build_fixed_batch_bank",
        "primary_150",
        "Rapid acquisition under a locked 150-update budget",
    )
    return all(marker in text for marker in required_markers)


def auto_find_main_code(explicit_path=None):
    """Resolve the full V4 source from common project layouts."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Specified main experiment does not exist: {path}")
        if not looks_like_locked_v4_source(path):
            raise RuntimeError(
                f"The specified file is not the locked full V4 experiment: {path}"
            )
        return path

    roots = []
    for root in (SCRIPT_DIR, Path.cwd().resolve()):
        if root not in roots:
            roots.append(root)

    # First use exact locked-source filenames in deterministic priority order.
    for name in LOCKED_V4_MAIN_NAMES:
        for root in roots:
            direct_candidates = (root / name, root / "upload" / name)
            for candidate in direct_candidates:
                if candidate.is_file() and candidate.resolve() != Path(__file__).resolve():
                    if looks_like_locked_v4_source(candidate):
                        return candidate.resolve()

    # Then search shallow project subdirectories, still requiring all V4 markers.
    candidates = []
    ignored_parts = {
        ".git", ".idea", ".venv", "venv", "env", "__pycache__", "site-packages"
    }
    for root in roots:
        for candidate in root.rglob("*.py"):
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) > 4 or any(part in ignored_parts for part in relative.parts):
                continue
            if candidate.resolve() == Path(__file__).resolve():
                continue
            if looks_like_locked_v4_source(candidate):
                candidates.append(candidate.resolve())
    candidates = sorted(set(candidates), key=lambda path: (len(path.parts), str(path)))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        formatted = "\n".join(f"  - {path}" for path in candidates)
        raise RuntimeError(
            "Multiple full V4 main-experiment files were found. Keep only the "
            "locked source beside this script, or pass --main-code explicitly:\n"
            f"{formatted}"
        )
    raise FileNotFoundError(
        "Could not find the locked full V4 main experiment. Put "
        "HDGM_v4_rapid_acquisition.py (or the original UUID-named source) "
        "in the same project as this script."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-checkpoint 3-module x 3-stage semantically aligned "
            "HDGM causal perturbation V2."
        )
    )
    parser.add_argument(
        "--main-code",
        type=Path,
        default=None,
        help="Optional override for the locked V4 main-experiment Python file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional override for the directory containing primary_150 checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults beside this script.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in FINAL_SEEDS),
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(PERTURBATION_MODES),
        help="Comma-separated subset of operation_cut,gate_trial_shuffle.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=150,
        help="Locked model snapshot used for the analysis.",
    )
    parser.add_argument(
        "--readout-width",
        type=int,
        default=4,
        help="Number of recurrent samples in the terminal readout window.",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=None,
        help="Override the configured 50 fixed test batches.",
    )
    parser.add_argument(
        "--analysis-batches",
        type=int,
        default=None,
        help="Override the configured 5 fixed analysis batches.",
    )
    parser.add_argument(
        "--skip-representation",
        action="store_true",
        help="Export behavior only; skip history and decision decoding.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="One-seed, two-batch pipeline audit; not a paper result.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def materialize_embedded_main():
    """Restore the exact locked V4 source into an automatic temporary location."""
    raw = zlib.decompress(base64.b85decode(EMBEDDED_V4_SOURCE_B85.encode("ascii")))
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EMBEDDED_V4_SHA256:
        raise RuntimeError(
            f"Embedded V4 source integrity check failed: {actual} != {EMBEDDED_V4_SHA256}"
        )
    cache_dir = Path(tempfile.gettempdir()) / "hdgm_v4_stage_perturbation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "HDGM_v4_rapid_acquisition_embedded.py"
    if not target.is_file() or sha256_file(target) != EMBEDDED_V4_SHA256:
        target.write_bytes(raw)
    return target.resolve()


def load_main_module(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = "hdgm_locked_v4_main"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import main experiment: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    required = (
        "ExperimentConfig",
        "SharedPathFourContextEnv",
        "HDGM",
        "build_fixed_batch_bank",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Main experiment is missing required names: {missing}")
    return module


def trusted_torch_load(path, map_location="cpu"):
    """Load only checkpoints created by the user's locked experiment."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def discover_checkpoints(checkpoint_dir, requested_seeds):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(checkpoint_dir)
    found = {}
    for path in sorted(checkpoint_dir.glob("*.pt")):
        checkpoint = trusted_torch_load(path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            continue
        seed = checkpoint.get("seed")
        mode = checkpoint.get("mode", "")
        max_updates = checkpoint.get("max_updates")
        if seed is None or int(seed) not in requested_seeds:
            continue
        if mode and mode != "primary_150":
            continue
        if max_updates is not None and int(max_updates) < 150:
            continue
        if int(seed) in found:
            raise RuntimeError(f"Multiple matching checkpoints found for seed {seed}.")
        found[int(seed)] = (path, checkpoint)
    missing = sorted(set(requested_seeds) - set(found))
    if missing:
        raise FileNotFoundError(f"Missing primary checkpoint(s) for seeds: {missing}")
    return found


def checkpoint_seed_inventory(checkpoint_dir, requested_seeds):
    """Return usable primary_150 seeds in one directory without failing on extras."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return set()
    usable = set()
    for path in sorted(checkpoint_dir.glob("*.pt")):
        try:
            checkpoint = trusted_torch_load(path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(checkpoint, dict):
            continue
        seed = checkpoint.get("seed")
        mode = checkpoint.get("mode", "")
        max_updates = checkpoint.get("max_updates")
        if seed is None:
            continue
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            continue
        if seed not in requested_seeds:
            continue
        if mode and mode != "primary_150":
            continue
        if max_updates is not None and int(max_updates) < 150:
            continue
        usable.add(seed)
    return usable


def auto_find_checkpoint_dir(explicit_path, requested_seeds):
    """Find one directory containing the complete locked ten-seed checkpoint set."""
    requested_seeds = set(int(seed) for seed in requested_seeds)
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Specified checkpoint directory does not exist: {path}")
        found = checkpoint_seed_inventory(path, requested_seeds)
        missing = sorted(requested_seeds - found)
        if missing:
            raise FileNotFoundError(
                f"The specified checkpoint directory is missing primary seeds: {missing}"
            )
        return path

    roots = []
    for root in (SCRIPT_DIR, Path.cwd().resolve()):
        if root not in roots and root.is_dir():
            roots.append(root)

    priority = []
    for root in roots:
        priority.extend(
            (
                root / "checkpoints",
                root / "paper_rapid_v4_outputs" / "checkpoints",
                root / "output" / "checkpoints",
                root / "upload",
                root,
            )
        )
    checked = set()
    partial = []
    for candidate in priority:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        seeds = checkpoint_seed_inventory(candidate, requested_seeds)
        if seeds == requested_seeds:
            return candidate
        if seeds:
            partial.append((candidate, seeds))

    # Fallback: shallow recursive search for folders that actually contain .pt files.
    ignored_parts = {
        ".git", ".idea", ".venv", "venv", "env", "__pycache__", "site-packages"
    }
    candidate_dirs = set()
    for root in roots:
        for checkpoint_path in root.rglob("*.pt"):
            try:
                relative = checkpoint_path.relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) > 6 or any(part in ignored_parts for part in relative.parts):
                continue
            candidate_dirs.add(checkpoint_path.parent.resolve())
    for candidate in sorted(candidate_dirs, key=lambda path: (len(path.parts), str(path))):
        if candidate in checked:
            continue
        seeds = checkpoint_seed_inventory(candidate, requested_seeds)
        if seeds == requested_seeds:
            return candidate
        if seeds:
            partial.append((candidate, seeds))

    details = ""
    if partial:
        best_path, best_seeds = max(partial, key=lambda item: len(item[1]))
        details = (
            f" Best partial directory: {best_path}; found seeds "
            f"{sorted(best_seeds)}; missing {sorted(requested_seeds - best_seeds)}."
        )
    raise FileNotFoundError(
        "Could not find one directory containing all ten locked primary_150 "
        f"checkpoints.{details} Put the ten .pt files in a 'checkpoints' folder "
        "beside this script."
    )


def resolve_checkpoint_dir_with_picker(explicit_path, requested_seeds):
    """Autodetect checkpoints or let a desktop user select their existing folder."""
    try:
        return auto_find_checkpoint_dir(explicit_path, requested_seeds)
    except FileNotFoundError as original_error:
        if explicit_path is not None:
            raise
        print("\nThe checkpoint folder was not found automatically.")
        print(
            "A folder-selection window will open; choose the existing folder "
            "containing the ten .pt files."
        )
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(
                title="Select the folder containing the 10 HDGM primary_150 checkpoints"
            )
            root.destroy()
        except Exception as gui_error:
            print(f"Folder picker unavailable: {gui_error}")
            try:
                selected = input(
                    "Paste the existing checkpoint-folder path here, then press Enter: "
                ).strip().strip('"')
            except (EOFError, KeyboardInterrupt) as input_error:
                raise FileNotFoundError(
                    f"{original_error}\nThe checkpoint folder could not be selected."
                ) from input_error
        if not selected:
            raise FileNotFoundError(
                f"{original_error}\nNo checkpoint folder was selected."
            )
        return auto_find_checkpoint_dir(Path(selected), requested_seeds)


def config_from_checkpoint(main_module, checkpoint):
    raw = dict(checkpoint["config"])
    return main_module.ExperimentConfig(**raw)


def build_environment(main_module, cfg, checkpoint, seed):
    env = main_module.SharedPathFourContextEnv(
        track_len=cfg.track_len,
        obs_dim=cfg.obs_dim,
        cue_width=cfg.cue_width,
        cueA_start_range=cfg.cueA_start_range,
        delay1_range=cfg.delay1_range,
        delay2_range=cfg.delay2_range,
        n_cues=cfg.n_cues,
        alias_prob=cfg.alias_prob,
        noise_std=cfg.noise_std,
        seed=seed,
    )
    maps = checkpoint["environment_maps"]
    env.pos_map = np.asarray(maps["pos_map"], dtype=np.float32).copy()
    env.cueA_map = np.asarray(maps["cueA_map"], dtype=np.float32).copy()
    env.cueB_map = np.asarray(maps["cueB_map"], dtype=np.float32).copy()
    return env


def build_locked_banks(main_module, env, cfg, checkpoint, test_batches, analysis_batches):
    split = checkpoint["split_seeds"]
    normal = main_module.build_fixed_batch_bank(
        env,
        n_batches=test_batches,
        batch_size=cfg.test_batch_size,
        delay_mode="normal",
        seed=int(split["test_normal"]),
    )
    long = main_module.build_fixed_batch_bank(
        env,
        n_batches=test_batches,
        batch_size=cfg.test_batch_size,
        delay_mode="long",
        seed=int(split["test_long"]),
    )
    analysis = main_module.build_fixed_batch_bank(
        env,
        n_batches=analysis_batches,
        batch_size=cfg.analysis_batch_size,
        delay_mode="normal",
        seed=int(split["analysis"]),
    )
    return {"normal": normal, "long": long}, analysis


def make_intervenable_hdgm_class(main_module):
    class IntervenableHDGM(main_module.HDGM):
        """Exact V4 forward plus test-only, equation-matched causal controls."""

        def set_gate_reference(self, gate_reference):
            self._gate_reference = {
                key: value.detach().clone().to(next(self.parameters()).device)
                for key, value in gate_reference.items()
            }

        @staticmethod
        def _stage_mask(phase_t, t, decision_positions, stage, readout_width):
            decision_positions = decision_positions.to(phase_t.device)
            readout_start = decision_positions - int(readout_width) + 1
            readout_mask = (t >= readout_start) & (t <= decision_positions)
            if stage == "encoding":
                return ((phase_t == 1) | (phase_t == 3)) & (~readout_mask)
            if stage == "maintenance":
                return ((phase_t == 2) | (phase_t == 4)) & (~readout_mask)
            if stage == "readout":
                return readout_mask
            raise ValueError(f"Unknown intervention stage: {stage}")

        def _apply_gate(self, gate, module_name, mask, intervention):
            if intervention is None or intervention.get("module") != module_name:
                return gate
            mode = intervention["mode"]
            if mode in ("operation_cut", CA1_PREDICTIVE_CUT_MODE):
                # Operation cuts act on the equation term controlled by the
                # module, not on the gate value itself.
                return gate
            if mode != "gate_trial_shuffle":
                raise ValueError(f"Unknown intervention mode: {mode}")
            if gate.shape[0] < 2:
                raise RuntimeError(
                    "gate_trial_shuffle requires a batch with at least two trials."
                )
            replacement = torch.roll(gate, shifts=1, dims=0)
            return torch.where(mask.unsqueeze(1), replacement, gate)

        def forward(
            self,
            x,
            a,
            cueA_zones=None,
            cueB_zones=None,
            decision_positions=None,
            return_details=False,
            skip_readout=False,
            phase=None,
            intervention=None,
            readout_width=4,
        ):
            if intervention is not None:
                if phase is None or decision_positions is None:
                    raise ValueError(
                        "Intervention scheduling requires phase and decision_positions."
                    )

            B, T, _ = x.shape
            g = torch.zeros(B, self.hidden_dim, device=x.device)
            M = torch.zeros(B, self.hidden_dim, self.hidden_dim, device=x.device)
            v = torch.zeros(B, self.fb_dim, device=x.device)
            decision_tensor = (
                torch.as_tensor(decision_positions, dtype=torch.long, device=x.device)
                if decision_positions is not None
                else None
            )

            pred_loss = 0.0
            mem_list, xhat_list, g_list = [], [], []
            ec_list, ca3_list, ca1_list = [], [], []
            write_strength_list, v_list = [], []
            pos_logits_list, phase_logits_list = [], []
            intervention_mask_list = []

            for t in range(T - 1):
                x_t = x[:, t]
                a_t = a[:, t]
                if intervention is None:
                    active_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
                else:
                    active_mask = self._stage_mask(
                        phase[:, t],
                        t,
                        decision_tensor,
                        intervention["stage"],
                        readout_width,
                    )
                operation_cut = (
                    intervention is not None
                    and intervention["mode"] == "operation_cut"
                )
                cut_ec = operation_cut and intervention["module"] == "ec"
                cut_ca3 = operation_cut and intervention["module"] == "ca3"
                cut_ca1 = operation_cut and intervention["module"] == "ca1"
                cut_ca1_prediction = (
                    intervention is not None
                    and intervention["mode"] == CA1_PREDICTIVE_CUT_MODE
                    and intervention["module"] == "ca1"
                )

                gate_ec = torch.sigmoid(
                    self.gate_ec(torch.cat([x_t, g, v], dim=1))
                )
                gate_ec = self._apply_gate(gate_ec, "ec", active_mask, intervention)
                x_mod = gate_ec * x_t
                if cut_ec:
                    x_mod = torch.where(
                        active_mask.unsqueeze(1),
                        torch.zeros_like(x_mod),
                        x_mod,
                    )
                x_emb = torch.tanh(self.fc_x(x_mod))

                a_emb = self.fc_a(self.action_emb(a_t))
                g_pred = torch.tanh(self.fc_g(g) + a_emb)

                h = x_emb
                ca3_steps = []
                for _ in range(5):
                    gate_ca3 = torch.sigmoid(
                        self.gate_ca3(torch.cat([h, g], dim=1))
                    )
                    gate_ca3 = self._apply_gate(
                        gate_ca3, "ca3", active_mask, intervention
                    )
                    associative_term = torch.bmm(M, h.unsqueeze(2)).squeeze(2)
                    if cut_ca3:
                        associative_term = torch.where(
                            active_mask.unsqueeze(1),
                            torch.zeros_like(associative_term),
                            associative_term,
                        )
                    recur = torch.tanh(0.5 * h + associative_term)
                    h = (1.0 - gate_ca3) * h + gate_ca3 * recur
                    ca3_steps.append(gate_ca3)

                p_t = h
                g = g_pred + self.alpha * p_t

                predictive_term = 0.5 * g
                if cut_ca1_prediction:
                    predictive_term = torch.where(
                        active_mask.unsqueeze(1),
                        torch.zeros_like(predictive_term),
                        predictive_term,
                    )
                memory_term = torch.bmm(M, g.unsqueeze(2)).squeeze(2)
                if cut_ca1:
                    memory_term = torch.where(
                        active_mask.unsqueeze(1),
                        torch.zeros_like(memory_term),
                        memory_term,
                    )
                p_hat = torch.tanh(predictive_term + memory_term)
                gain_ca1 = torch.sigmoid(
                    self.gate_ca1(torch.cat([g, p_hat, v], dim=1))
                )
                gain_ca1 = self._apply_gate(
                    gain_ca1, "ca1", active_mask, intervention
                )
                p_out = torch.relu(
                    p_hat * (self.c1 + self.c2 * gain_ca1) - self.beta
                )

                x_hat = self.decoder(p_out)
                pos_logits = self.pos_head(p_out)
                phase_logits = self.phase_head(p_out)
                pred_loss = pred_loss + ((x_hat - x[:, t + 1]) ** 2).mean()

                v = self.rho * v + self.fb_proj(p_out)
                write_strength = gate_ec.mean(dim=1, keepdim=True)
                if cut_ec:
                    write_strength = torch.where(
                        active_mask.unsqueeze(1),
                        torch.zeros_like(write_strength),
                        write_strength,
                    )
                outer = torch.bmm(p_t.unsqueeze(2), p_t.unsqueeze(1))
                M = self.lambda_m * M + write_strength.unsqueeze(-1) * outer

                mem_list.append(p_out)
                xhat_list.append(x_hat)
                g_list.append(g)
                ec_list.append(gate_ec)
                ca3_list.append(torch.stack(ca3_steps, dim=1).mean(dim=1))
                ca1_list.append(gain_ca1)
                write_strength_list.append(write_strength.squeeze(1))
                v_list.append(v)
                pos_logits_list.append(pos_logits)
                phase_logits_list.append(phase_logits)
                intervention_mask_list.append(active_mask)

            mem = torch.stack(mem_list, dim=1)
            xhat = torch.stack(xhat_list, dim=1)
            g_hist = torch.stack(g_list, dim=1)
            ec_hist = torch.stack(ec_list, dim=1)
            ca3_hist = torch.stack(ca3_list, dim=1)
            ca1_hist = torch.stack(ca1_list, dim=1)
            write_hist = torch.stack(write_strength_list, dim=1)
            v_hist = torch.stack(v_list, dim=1)
            pos_logits_hist = torch.stack(pos_logits_list, dim=1)
            phase_logits_hist = torch.stack(phase_logits_list, dim=1)
            intervention_mask_hist = torch.stack(intervention_mask_list, dim=1)

            if skip_readout:
                logits = None
                readout_out = None
            elif return_details and hasattr(self.readout, "cueA_pool"):
                readout_out = self.readout(
                    mem,
                    cueA_zones,
                    cueB_zones,
                    decision_positions,
                    return_details=True,
                )
                logits = readout_out["logits"]
            else:
                logits = self.readout(
                    mem, cueA_zones, cueB_zones, decision_positions
                )
                readout_out = None

            cueA_aux_logits = None
            if cueA_zones is not None and self.cueA_aux_head is not None:
                cueA_reprs = [
                    mem[b, a0:a1, :].mean(dim=0)
                    for b, (a0, a1) in enumerate(cueA_zones)
                ]
                cueA_aux_logits = self.cueA_aux_head(torch.stack(cueA_reprs))

            cueB_aux_logits = None
            if cueB_zones is not None and self.cueB_aux_head is not None:
                cueB_reprs = [
                    mem[b, b0:b1, :].mean(dim=0)
                    for b, (b0, b1) in enumerate(cueB_zones)
                ]
                cueB_aux_logits = self.cueB_aux_head(torch.stack(cueB_reprs))

            if return_details:
                out = {
                    "pred_loss": pred_loss,
                    "logits": logits,
                    "mem": mem,
                    "xhat": xhat,
                    "g_hist": g_hist,
                    "ec_hist": ec_hist,
                    "ca3_hist": ca3_hist,
                    "ca1_hist": ca1_hist,
                    "write_hist": write_hist,
                    "v_hist": v_hist,
                    "pos_logits_hist": pos_logits_hist,
                    "phase_logits_hist": phase_logits_hist,
                    "cueA_aux_logits": cueA_aux_logits,
                    "cueB_aux_logits": cueB_aux_logits,
                    "intervention_mask_hist": intervention_mask_hist,
                }
                if readout_out is not None:
                    for key in (
                        "cueA_attn",
                        "cueB_attn",
                        "cueA_repr",
                        "cueB_repr",
                        "decision_repr",
                    ):
                        out[key] = readout_out[key]
                return out
            return pred_loss, logits

    return IntervenableHDGM


def instantiate_model(main_module, cfg, env, checkpoint, checkpoint_step, device):
    model_class = make_intervenable_hdgm_class(main_module)
    model = model_class(
        obs_dim=cfg.obs_dim,
        hidden_dim=96,
        track_len=env.track_len,
        action_dim=env.action_dim,
        n_classes=4,
        cfg=cfg,
    ).to(device)
    snapshots = checkpoint["snapshots"]
    if checkpoint_step not in snapshots:
        raise KeyError(
            f"Checkpoint step {checkpoint_step} is unavailable; "
            f"found {sorted(snapshots)}"
        )
    state = snapshots[checkpoint_step].get("hdgm")
    if state is None:
        state = snapshots[checkpoint_step].get("gate")
    if state is None:
        raise KeyError("Snapshot contains neither 'hdgm' nor 'gate' state.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def unpack_batch(batch):
    return {
        "x": batch[0],
        "a": batch[1],
        "pos": batch[2],
        "phase": batch[3],
        "labels": batch[4],
        "cueA": batch[5],
        "cueB": batch[6],
        "cueA_zones": batch[7],
        "cueB_zones": batch[8],
        "decision_positions": batch[9],
        "delay1s": batch[10],
        "delay2s": batch[11],
    }


def forward_batch(model, batch, device, intervention, readout_width, skip_readout=False):
    item = unpack_batch(batch)
    return model(
        item["x"].to(device),
        item["a"].to(device),
        cueA_zones=None if skip_readout else item["cueA_zones"],
        cueB_zones=None if skip_readout else item["cueB_zones"],
        decision_positions=item["decision_positions"],
        return_details=True,
        skip_readout=skip_readout,
        phase=item["phase"].to(device),
        intervention=intervention,
        readout_width=readout_width,
    )


def validate_control_equivalence(
    main_module, cfg, env, intervenable_model, batch, device, readout_width
):
    """Verify that the added intervention interface changes nothing in control."""
    reference_model = main_module.HDGM(
        obs_dim=cfg.obs_dim,
        hidden_dim=96,
        track_len=env.track_len,
        action_dim=env.action_dim,
        n_classes=4,
        cfg=cfg,
    ).to(device)
    reference_model.load_state_dict(intervenable_model.state_dict(), strict=True)
    reference_model.eval()
    item = unpack_batch(batch)
    with torch.no_grad():
        reference = reference_model(
            item["x"].to(device),
            item["a"].to(device),
            cueA_zones=item["cueA_zones"],
            cueB_zones=item["cueB_zones"],
            decision_positions=item["decision_positions"],
            return_details=True,
        )
        candidate = forward_batch(
            intervenable_model,
            batch,
            device,
            intervention=None,
            readout_width=readout_width,
            skip_readout=False,
        )
    checked_keys = (
        "logits", "mem", "g_hist", "ec_hist", "ca3_hist", "ca1_hist",
        "write_hist", "v_hist",
    )
    errors = {
        key: float((reference[key] - candidate[key]).abs().max().item())
        for key in checked_keys
    }
    overall = max(errors.values())
    if overall > 1e-7:
        raise RuntimeError(
            f"Control-equivalence validation failed: max_abs_error={overall}"
        )
    del reference_model
    return {"max_abs_error": overall, "by_tensor": errors}


def calibrate_gate_reference(model, analysis_bank, device, readout_width):
    sums = {
        "ec": torch.zeros(model.obs_dim, dtype=torch.float64, device=device),
        "ca3": torch.zeros(model.hidden_dim, dtype=torch.float64, device=device),
        "ca1": torch.zeros(model.hidden_dim, dtype=torch.float64, device=device),
    }
    counts = {key: 0 for key in sums}
    key_map = {"ec": "ec_hist", "ca3": "ca3_hist", "ca1": "ca1_hist"}
    with torch.no_grad():
        for batch in analysis_bank:
            out = forward_batch(
                model, batch, device, intervention=None,
                readout_width=readout_width, skip_readout=True,
            )
            for module_name, history_key in key_map.items():
                history = out[history_key].double()
                sums[module_name] += history.sum(dim=(0, 1))
                counts[module_name] += history.shape[0] * history.shape[1]
    reference = {
        module_name: (sums[module_name] / counts[module_name]).float()
        for module_name in sums
    }
    model.set_gate_reference(reference)
    return {
        key: value.detach().cpu().numpy().tolist()
        for key, value in reference.items()
    }


def confusion_from_predictions(labels, predictions, n_classes=4):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, predicted in zip(labels, predictions):
        matrix[int(truth), int(predicted)] += 1
    return matrix


def metrics_from_confusion(matrix):
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else float("nan")
    class_accuracy, f1_values = [], []
    for class_id in range(matrix.shape[0]):
        tp = float(matrix[class_id, class_id])
        fn = float(matrix[class_id, :].sum() - tp)
        fp = float(matrix[:, class_id].sum() - tp)
        class_total = tp + fn
        class_accuracy.append(float(tp / class_total) if class_total else None)
        denominator = 2.0 * tp + fp + fn
        f1_values.append(float(2.0 * tp / denominator) if denominator else 0.0)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "class_accuracy": class_accuracy,
        "confusion_matrix": matrix.tolist(),
        "n_trials": total,
    }


def evaluate_behavior(model, bank, device, intervention, readout_width):
    labels_all, predictions_all = [], []
    intervened_steps = 0
    possible_steps = 0
    with torch.no_grad():
        for batch in bank:
            item = unpack_batch(batch)
            out = forward_batch(
                model, batch, device, intervention,
                readout_width=readout_width, skip_readout=False,
            )
            predictions = out["logits"].argmax(dim=1).detach().cpu().numpy()
            labels = item["labels"].detach().cpu().numpy()
            labels_all.append(labels)
            predictions_all.append(predictions)
            mask = out["intervention_mask_hist"]
            intervened_steps += int(mask.sum().item())
            possible_steps += int(mask.numel())
    labels_all = np.concatenate(labels_all)
    predictions_all = np.concatenate(predictions_all)
    matrix = confusion_from_predictions(labels_all, predictions_all, N_CLASSES)
    result = metrics_from_confusion(matrix)
    result["intervened_time_steps"] = intervened_steps
    result["intervened_steps_per_trial"] = float(
        intervened_steps / max(1, len(labels_all))
    )
    result["intervened_fraction"] = float(
        intervened_steps / max(1, possible_steps)
    )
    return result


def residualize_by_position(features, positions):
    features = np.asarray(features, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    residual = features.copy()
    for position in np.unique(positions):
        mask = positions == position
        residual[mask] -= residual[mask].mean(axis=0, keepdims=True)
    return residual


def stratified_ridge_accuracy(features, labels, seed):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    classifier = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=1.0),
    )
    scores = cross_val_score(classifier, features, labels, cv=splitter, n_jobs=1)
    return float(scores.mean())


def standardized_binary_distance(features, labels):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    scale = features.std(axis=0, ddof=0)
    scale[scale < 1e-6] = 1.0
    standardized = features / scale
    delta = standardized[labels == 1].mean(axis=0) - standardized[labels == 0].mean(axis=0)
    return float(np.linalg.norm(delta) / math.sqrt(features.shape[1]))


def evaluate_representations(model, bank, device, intervention, readout_width, seed):
    delay_states, delay_positions, delay_trial_ids = [], [], []
    cue_a_by_trial, class_by_trial, decision_states = [], [], []
    trial_offset = 0
    with torch.no_grad():
        for batch in bank:
            item = unpack_batch(batch)
            out = forward_batch(
                model, batch, device, intervention,
                readout_width=readout_width, skip_readout=True,
            )
            mem = out["mem"].detach().cpu().numpy()
            phase = item["phase"][:, :mem.shape[1]].numpy()
            pos = item["pos"][:, :mem.shape[1]].numpy()
            cue_a = item["cueA"].numpy()
            labels = item["labels"].numpy()
            for local_trial in range(mem.shape[0]):
                global_trial = trial_offset + local_trial
                indices = np.flatnonzero(phase[local_trial] == 2)
                if len(indices) == 0:
                    raise RuntimeError("A trial has no Delay-1 samples.")
                delay_states.append(mem[local_trial, indices])
                delay_positions.append(pos[local_trial, indices])
                delay_trial_ids.append(
                    np.full(len(indices), global_trial, dtype=np.int64)
                )
                decision_index = max(
                    0,
                    min(
                        int(item["decision_positions"][local_trial]) - 1,
                        mem.shape[1] - 1,
                    ),
                )
                decision_states.append(mem[local_trial, decision_index])
                cue_a_by_trial.append(int(cue_a[local_trial]))
                class_by_trial.append(int(labels[local_trial]))
            trial_offset += mem.shape[0]

    delay_states = np.concatenate(delay_states, axis=0)
    delay_positions = np.concatenate(delay_positions, axis=0)
    delay_trial_ids = np.concatenate(delay_trial_ids, axis=0)
    delay_residual = residualize_by_position(delay_states, delay_positions)
    n_trials = len(cue_a_by_trial)
    trial_features = np.zeros((n_trials, delay_residual.shape[1]), dtype=np.float64)
    for trial_id in range(n_trials):
        trial_features[trial_id] = delay_residual[delay_trial_ids == trial_id].mean(axis=0)

    cue_a_by_trial = np.asarray(cue_a_by_trial, dtype=np.int64)
    class_by_trial = np.asarray(class_by_trial, dtype=np.int64)
    decision_states = np.asarray(decision_states, dtype=np.float64)
    return {
        "delay1_cueA_decoding": stratified_ridge_accuracy(
            trial_features, cue_a_by_trial, seed + 101
        ),
        "delay1_state_distance": standardized_binary_distance(
            trial_features, cue_a_by_trial
        ),
        "decision_pre_class_decoding": stratified_ridge_accuracy(
            decision_states, class_by_trial, seed + 202
        ),
        "n_analysis_trials": int(n_trials),
    }


OPERATION_LABELS = {
    ("operation_cut", "ec"): "remove selected afference and EC write strength",
    ("operation_cut", "ca3"): "remove associative recurrent term M@h",
    ("operation_cut", "ca1"): "remove memory-derived integration term M@g",
    ("gate_trial_shuffle", "ec"): "shuffle EC gate across trials",
    ("gate_trial_shuffle", "ca3"): "shuffle CA3 gate across trials",
    ("gate_trial_shuffle", "ca1"): "shuffle CA1 gate across trials",
    (CA1_PREDICTIVE_CUT_MODE, "ca1"): "remove predictive integration term 0.5*g",
}


def intervention_specs(modes):
    yield {
        "condition": "control",
        "mode": "control",
        "module": None,
        "stage": None,
        "operation_target": "none",
        "intervention": None,
    }
    for mode, module_name, stage in itertools.product(modes, MODULES, STAGES):
        yield {
            "condition": f"{mode}__{module_name}__{stage}",
            "mode": mode,
            "module": module_name,
            "stage": stage,
            "operation_target": OPERATION_LABELS[(mode, module_name)],
            "intervention": {
                "mode": mode,
                "module": module_name,
                "stage": stage,
            },
        }
    if "operation_cut" in modes:
        for stage in STAGES:
            yield {
                "condition": f"{CA1_PREDICTIVE_CUT_MODE}__ca1__{stage}",
                "mode": CA1_PREDICTIVE_CUT_MODE,
                "module": "ca1",
                "stage": stage,
                "operation_target": OPERATION_LABELS[(CA1_PREDICTIVE_CUT_MODE, "ca1")],
                "intervention": {
                    "mode": CA1_PREDICTIVE_CUT_MODE,
                    "module": "ca1",
                    "stage": stage,
                },
            }


def exact_sign_flip_test(differences):
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    n = len(differences)
    exceed = 0
    total = 2 ** n
    for bits in range(total):
        signs = np.asarray(
            [1.0 if (bits >> index) & 1 else -1.0 for index in range(n)]
        )
        statistic = abs(float((differences * signs).mean()))
        if statistic >= observed - 1e-15:
            exceed += 1
    sd = float(differences.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean_difference": float(differences.mean()),
        "p_two_sided_exact": float(exceed / total),
        "cohen_dz": float(differences.mean() / (sd + 1e-12)),
        "n": int(n),
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


STATISTIC_METRICS = (
    "normal_accuracy",
    "normal_macro_f1",
    "long_accuracy",
    "long_macro_f1",
    "delay1_cueA_decoding",
    "delay1_state_distance",
    "decision_pre_class_decoding",
)
TARGET_STAGE = {"ec": "encoding", "ca3": "maintenance", "ca1": "readout"}


def condition_metric(condition, metric):
    if metric == "normal_accuracy":
        return condition["normal"]["accuracy"]
    if metric == "normal_macro_f1":
        return condition["normal"]["macro_f1"]
    if metric == "long_accuracy":
        return condition["long"]["accuracy"]
    if metric == "long_macro_f1":
        return condition["long"]["macro_f1"]
    return condition.get("representation", {}).get(metric)


def paired_drops(records, condition_id, metric):
    drops = []
    for record in records:
        control_value = condition_metric(record["conditions"]["control"], metric)
        condition_value = condition_metric(record["conditions"][condition_id], metric)
        if control_value is None or condition_value is None:
            return None
        drops.append(float(control_value - condition_value))
    return np.asarray(drops, dtype=np.float64)


def add_holm(tests, family_size_label):
    corrected = holm_adjust([test["p_two_sided_exact"] for test in tests])
    for test, adjusted in zip(tests, corrected):
        test[family_size_label] = float(adjusted)


def aggregate_statistics(records, modes):
    """Confirmatory tests for cell damage and the two preregistered contrasts."""
    output = {}
    for mode in modes:
        mode_output = {
            "condition_tests": {},
            "targeted_within_module_contrasts": {},
            "targeted_between_module_contrasts": {},
        }
        for metric in STATISTIC_METRICS:
            condition_ids = [
                f"{mode}__{module_name}__{stage}"
                for module_name, stage in itertools.product(MODULES, STAGES)
            ]
            cell_drops = {
                condition_id: paired_drops(records, condition_id, metric)
                for condition_id in condition_ids
            }
            if any(value is None for value in cell_drops.values()):
                continue

            cell_tests = [
                exact_sign_flip_test(cell_drops[condition_id])
                for condition_id in condition_ids
            ]
            add_holm(cell_tests, "holm_p_across_9_conditions")
            mode_output["condition_tests"][metric] = {
                condition_id: test
                for condition_id, test in zip(condition_ids, cell_tests)
            }

            within_tests = []
            between_tests = []
            for module_name in MODULES:
                target_stage = TARGET_STAGE[module_name]
                off_stages = [stage for stage in STAGES if stage != target_stage]
                target = cell_drops[f"{mode}__{module_name}__{target_stage}"]
                off_mean = np.mean(
                    [cell_drops[f"{mode}__{module_name}__{stage}"] for stage in off_stages],
                    axis=0,
                )
                within_test = exact_sign_flip_test(target - off_mean)
                within_test.update({
                    "module": module_name,
                    "target_stage": target_stage,
                    "contrast_definition": "target-stage damage minus mean off-stage damage",
                })
                within_tests.append(within_test)

                other_modules = [name for name in MODULES if name != module_name]
                other_mean = np.mean(
                    [cell_drops[f"{mode}__{name}__{target_stage}"] for name in other_modules],
                    axis=0,
                )
                between_test = exact_sign_flip_test(target - other_mean)
                between_test.update({
                    "module": module_name,
                    "target_stage": target_stage,
                    "contrast_definition": "target-module damage minus mean other-module damage at target stage",
                })
                between_tests.append(between_test)

            add_holm(within_tests, "holm_p_across_3_modules")
            add_holm(between_tests, "holm_p_across_3_modules")
            mode_output["targeted_within_module_contrasts"][metric] = {
                test["module"]: test for test in within_tests
            }
            mode_output["targeted_between_module_contrasts"][metric] = {
                test["module"]: test for test in between_tests
            }
        output[mode] = mode_output

    # CA1 component decomposition: memory-derived M@g versus predictive 0.5*g.
    if "operation_cut" in modes:
        component_output = {}
        for metric in STATISTIC_METRICS:
            condition_ids = []
            component_labels = []
            tests = []
            for component_mode, component_label in (
                ("operation_cut", "memory_derived_Mg"),
                (CA1_PREDICTIVE_CUT_MODE, "predictive_0.5g"),
            ):
                for stage in STAGES:
                    condition_id = f"{component_mode}__ca1__{stage}"
                    drops = paired_drops(records, condition_id, metric)
                    if drops is None:
                        tests = []
                        break
                    condition_ids.append(condition_id)
                    component_labels.append(component_label)
                    test = exact_sign_flip_test(drops)
                    test.update({"component": component_label, "stage": stage})
                    tests.append(test)
                if not tests:
                    break
            if not tests:
                continue
            add_holm(tests, "holm_p_across_6_component_stage_conditions")
            component_output[metric] = {
                condition_id: test
                for condition_id, test in zip(condition_ids, tests)
            }
        output["ca1_component_decomposition"] = component_output
    return output


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def atomic_json_dump(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_csv(records, path):
    rows = []
    for record in records:
        control = record["conditions"]["control"]
        for condition_name, condition in record["conditions"].items():
            normal = condition["normal"]
            long_result = condition["long"]
            representation = condition.get("representation", {})
            control_representation = control.get("representation", {})
            rows.append({
                "seed": record["seed"],
                "condition": condition_name,
                "mode": condition["mode"],
                "module": condition["module"],
                "stage": condition["stage"],
                "operation_target": condition.get("operation_target"),
                "normal_accuracy": normal["accuracy"],
                "normal_macro_f1": normal["macro_f1"],
                "long_accuracy": long_result["accuracy"],
                "long_macro_f1": long_result["macro_f1"],
                "normal_accuracy_drop": control["normal"]["accuracy"] - normal["accuracy"],
                "long_accuracy_drop": control["long"]["accuracy"] - long_result["accuracy"],
                "normal_macro_f1_drop": control["normal"]["macro_f1"] - normal["macro_f1"],
                "long_macro_f1_drop": control["long"]["macro_f1"] - long_result["macro_f1"],
                "intervened_steps_per_trial_normal": normal["intervened_steps_per_trial"],
                "intervened_fraction_normal": normal["intervened_fraction"],
                "delay1_cueA_decoding": representation.get("delay1_cueA_decoding"),
                "delay1_state_distance": representation.get("delay1_state_distance"),
                "decision_pre_class_decoding": representation.get("decision_pre_class_decoding"),
                "delay1_cueA_decoding_drop": (
                    control_representation.get("delay1_cueA_decoding")
                    - representation.get("delay1_cueA_decoding")
                    if representation and control_representation else None
                ),
                "delay1_state_distance_drop": (
                    control_representation.get("delay1_state_distance")
                    - representation.get("delay1_state_distance")
                    if representation and control_representation else None
                ),
                "decision_pre_class_decoding_drop": (
                    control_representation.get("decision_pre_class_decoding")
                    - representation.get("decision_pre_class_decoding")
                    if representation and control_representation else None
                ),
            })
    fieldnames = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_npz(records, modes, path):
    seeds = np.asarray([record["seed"] for record in records], dtype=np.int64)
    shape = (len(records), len(modes), len(MODULES), len(STAGES))
    normal_drop = np.full(shape, np.nan, dtype=np.float64)
    long_drop = np.full(shape, np.nan, dtype=np.float64)
    macro_f1_drop = np.full(shape, np.nan, dtype=np.float64)
    long_macro_f1_drop = np.full(shape, np.nan, dtype=np.float64)
    history_decode_drop = np.full(shape, np.nan, dtype=np.float64)
    state_distance_drop = np.full(shape, np.nan, dtype=np.float64)
    decision_decode_drop = np.full(shape, np.nan, dtype=np.float64)
    step_count = np.full(shape, np.nan, dtype=np.float64)
    ca1_predictive_shape = (len(records), len(STAGES))
    ca1_predictive_normal_drop = np.full(ca1_predictive_shape, np.nan, dtype=np.float64)
    ca1_predictive_long_drop = np.full(ca1_predictive_shape, np.nan, dtype=np.float64)
    ca1_predictive_history_drop = np.full(ca1_predictive_shape, np.nan, dtype=np.float64)
    ca1_predictive_state_drop = np.full(ca1_predictive_shape, np.nan, dtype=np.float64)
    ca1_predictive_decision_drop = np.full(ca1_predictive_shape, np.nan, dtype=np.float64)
    control_normal = np.empty(len(records), dtype=np.float64)
    control_long = np.empty(len(records), dtype=np.float64)

    for seed_index, record in enumerate(records):
        control = record["conditions"]["control"]
        control_normal[seed_index] = control["normal"]["accuracy"]
        control_long[seed_index] = control["long"]["accuracy"]
        control_rep = control.get("representation", {})
        for mode_index, mode in enumerate(modes):
            for module_index, module_name in enumerate(MODULES):
                for stage_index, stage in enumerate(STAGES):
                    condition = record["conditions"][
                        f"{mode}__{module_name}__{stage}"
                    ]
                    normal_drop[seed_index, mode_index, module_index, stage_index] = (
                        control["normal"]["accuracy"] - condition["normal"]["accuracy"]
                    )
                    long_drop[seed_index, mode_index, module_index, stage_index] = (
                        control["long"]["accuracy"] - condition["long"]["accuracy"]
                    )
                    macro_f1_drop[seed_index, mode_index, module_index, stage_index] = (
                        control["normal"]["macro_f1"] - condition["normal"]["macro_f1"]
                    )
                    long_macro_f1_drop[seed_index, mode_index, module_index, stage_index] = (
                        control["long"]["macro_f1"] - condition["long"]["macro_f1"]
                    )
                    step_count[seed_index, mode_index, module_index, stage_index] = (
                        condition["normal"]["intervened_steps_per_trial"]
                    )
                    rep = condition.get("representation", {})
                    if rep and control_rep:
                        history_decode_drop[seed_index, mode_index, module_index, stage_index] = (
                            control_rep["delay1_cueA_decoding"] - rep["delay1_cueA_decoding"]
                        )
                        state_distance_drop[seed_index, mode_index, module_index, stage_index] = (
                            control_rep["delay1_state_distance"] - rep["delay1_state_distance"]
                        )
                        decision_decode_drop[seed_index, mode_index, module_index, stage_index] = (
                            control_rep["decision_pre_class_decoding"]
                            - rep["decision_pre_class_decoding"]
                        )

        for stage_index, stage in enumerate(STAGES):
            condition_id = f"{CA1_PREDICTIVE_CUT_MODE}__ca1__{stage}"
            if condition_id not in record["conditions"]:
                continue
            condition = record["conditions"][condition_id]
            ca1_predictive_normal_drop[seed_index, stage_index] = (
                control["normal"]["accuracy"] - condition["normal"]["accuracy"]
            )
            ca1_predictive_long_drop[seed_index, stage_index] = (
                control["long"]["accuracy"] - condition["long"]["accuracy"]
            )
            rep = condition.get("representation", {})
            if rep and control_rep:
                ca1_predictive_history_drop[seed_index, stage_index] = (
                    control_rep["delay1_cueA_decoding"] - rep["delay1_cueA_decoding"]
                )
                ca1_predictive_state_drop[seed_index, stage_index] = (
                    control_rep["delay1_state_distance"] - rep["delay1_state_distance"]
                )
                ca1_predictive_decision_drop[seed_index, stage_index] = (
                    control_rep["decision_pre_class_decoding"]
                    - rep["decision_pre_class_decoding"]
                )

    np.savez_compressed(
        path,
        seeds=seeds,
        modes=np.asarray(modes),
        modules=np.asarray(MODULES),
        stages=np.asarray(STAGES),
        control_normal_accuracy=control_normal,
        control_long_accuracy=control_long,
        normal_accuracy_drop=normal_drop,
        long_accuracy_drop=long_drop,
        normal_macro_f1_drop=macro_f1_drop,
        long_macro_f1_drop=long_macro_f1_drop,
        delay1_cueA_decoding_drop=history_decode_drop,
        delay1_state_distance_drop=state_distance_drop,
        decision_pre_class_decoding_drop=decision_decode_drop,
        intervened_steps_per_trial=step_count,
        ca1_predictive_cut_normal_accuracy_drop=ca1_predictive_normal_drop,
        ca1_predictive_cut_long_accuracy_drop=ca1_predictive_long_drop,
        ca1_predictive_cut_delay1_cueA_decoding_drop=ca1_predictive_history_drop,
        ca1_predictive_cut_delay1_state_distance_drop=ca1_predictive_state_drop,
        ca1_predictive_cut_decision_pre_class_decoding_drop=ca1_predictive_decision_drop,
    )


def run_seed(
    seed,
    checkpoint_path,
    checkpoint,
    main_module,
    args,
    device,
    modes,
):
    print("\n" + "=" * 112)
    print(f"SEMANTIC CAUSAL PERTURBATION V2 | seed={seed} | device={device}")
    print("=" * 112)
    set_seed(seed)
    cfg = config_from_checkpoint(main_module, checkpoint)
    env = build_environment(main_module, cfg, checkpoint, seed)
    test_batches = 2 if args.quick else (args.test_batches or cfg.test_batches)
    analysis_batches = 2 if args.quick else (
        args.analysis_batches or cfg.analysis_batches
    )
    test_banks, analysis_bank = build_locked_banks(
        main_module,
        env,
        cfg,
        checkpoint,
        test_batches=test_batches,
        analysis_batches=analysis_batches,
    )
    model = instantiate_model(
        main_module,
        cfg,
        env,
        checkpoint,
        args.checkpoint_step,
        device,
    )
    equivalence = validate_control_equivalence(
        main_module,
        cfg,
        env,
        model,
        analysis_bank[0],
        device,
        args.readout_width,
    )
    print(
        "Control-equivalence max_abs_error="
        f"{equivalence['max_abs_error']:.3g}"
    )
    reference = calibrate_gate_reference(
        model, analysis_bank, device, args.readout_width
    )
    conditions = {}
    specs = list(intervention_specs(modes))
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index:02d}/{len(specs):02d}] "
            f"{spec['condition']}",
            flush=True,
        )
        normal = evaluate_behavior(
            model,
            test_banks["normal"],
            device,
            spec["intervention"],
            args.readout_width,
        )
        long_result = evaluate_behavior(
            model,
            test_banks["long"],
            device,
            spec["intervention"],
            args.readout_width,
        )
        condition = {
            "mode": spec["mode"],
            "module": spec["module"],
            "stage": spec["stage"],
            "operation_target": spec["operation_target"],
            "normal": normal,
            "long": long_result,
        }
        if not args.skip_representation:
            condition["representation"] = evaluate_representations(
                model,
                analysis_bank,
                device,
                spec["intervention"],
                args.readout_width,
                seed,
            )
        conditions[spec["condition"]] = condition
        print(
            f"    normal={normal['accuracy']:.4f}, "
            f"long={long_result['accuracy']:.4f}, "
            f"steps/trial={normal['intervened_steps_per_trial']:.2f}"
        )

    return {
        "seed": int(seed),
        "checkpoint_file": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(args.checkpoint_step),
        "config": asdict(cfg),
        "split_seeds": copy.deepcopy(checkpoint["split_seeds"]),
        "test_batches_per_condition": int(test_batches),
        "analysis_batches_per_condition": int(analysis_batches),
        "control_equivalence": equivalence,
        "gate_reference_per_unit": reference,
        "conditions": conditions,
    }


def main():
    args = parse_args()
    requested_seeds = tuple(
        int(value.strip()) for value in args.seeds.split(",") if value.strip()
    )
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    invalid_modes = sorted(set(modes) - set(PERTURBATION_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported modes: {invalid_modes}")
    if args.quick:
        requested_seeds = requested_seeds[:1]
    if args.readout_width < 1:
        raise ValueError("readout_width must be positive")

    # Zero-configuration resolution for direct PyCharm execution.
    if args.main_code is None:
        args.main_code = materialize_embedded_main()
    else:
        args.main_code = auto_find_main_code(args.main_code)
    args.checkpoint_dir = resolve_checkpoint_dir_with_picker(
        args.checkpoint_dir,
        requested_seeds,
    )
    if args.output_dir is None:
        args.output_dir = SCRIPT_DIR / "output" / "figure1d_data"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = (
        "hdgm_semantic_causal_perturbation_v2_quick_audit"
        if args.quick
        else "hdgm_semantic_causal_perturbation_v2"
    )
    output_json = args.output_dir / f"{output_stem}.json"
    output_csv = args.output_dir / f"{output_stem}.csv"
    output_npz = args.output_dir / f"{output_stem}.npz"
    partial_json = args.output_dir / f"{output_stem}.partial.json"

    device = choose_device(args.device)
    print("=" * 112)
    print("HDGM V4 SEMANTICALLY ALIGNED CAUSAL PERTURBATION V2")
    print("=" * 112)
    print(f"Main experiment : {args.main_code}")
    print(f"Checkpoints     : {args.checkpoint_dir}")
    print(f"Output          : {args.output_dir}")
    print(f"Device          : {device}")
    print(f"Seeds           : {requested_seeds}")
    print(f"Run type        : {'QUICK AUDIT' if args.quick else 'FULL CONFIRMATORY'}")
    print("=" * 112)
    main_module = load_main_module(args.main_code)
    main_module.device = device
    checkpoints = discover_checkpoints(args.checkpoint_dir, requested_seeds)

    protocol = {
        "experiment": "HDGM V4 frozen semantically aligned causal perturbation v2",
        "main_code": str(args.main_code.resolve()),
        "main_code_sha256": sha256_file(args.main_code),
        "checkpoint_step": int(args.checkpoint_step),
        "seeds": list(requested_seeds),
        "modules": list(MODULES),
        "stages": list(STAGES),
        "modes": list(modes),
        "readout_width": int(args.readout_width),
        "stage_definition": {
            "encoding": "phase 1 or 3 (Cue A or Cue B)",
            "maintenance": "phase 2 or 4 excluding terminal readout window",
            "readout": "terminal readout_width samples ending at decision",
        },
        "operation_cut_definition": {
            "ec": (
                "selected afferent x_mod and EC-controlled memory-write strength "
                "are set to zero at the targeted stage"
            ),
            "ca3": (
                "the associative recurrent term M@h is removed at the targeted "
                "stage; the learned gate, local term, and five iterations remain"
            ),
            "ca1": (
                "the memory-derived term M@g is removed at the targeted stage; "
                "the predictive term, learned gain, residual path, and feedback remain"
            ),
        },
        "gate_trial_shuffle_definition": (
            "the selected learned gate is cyclically permuted across trials at "
            "the same time step, preserving its population distribution while "
            "breaking trial alignment"
        ),
        "ca1_predictive_cut_definition": (
            "the predictive component 0.5*g is removed at each tested stage, "
            "providing a preregistered component-level counterpart to the CA1 "
            "memory-derived M@g cut"
        ),
        "weights_frozen": True,
        "no_retraining": True,
        "identical_fixed_banks_across_conditions": True,
        "phase_metadata_used_only_for_test_intervention_schedule": True,
        "quick_audit_not_for_paper": bool(args.quick),
    }

    records = []
    for seed in requested_seeds:
        path, checkpoint = checkpoints[seed]
        record = run_seed(
            seed,
            path,
            checkpoint,
            main_module,
            args,
            device,
            modes,
        )
        records.append(record)
        atomic_json_dump(
            {"protocol": protocol, "records": records}, partial_json
        )

    statistics = (
        aggregate_statistics(records, modes)
        if len(records) >= 2
        else {"note": "Quick audit has one seed; confirmatory statistics omitted."}
    )
    payload = {
        "protocol": protocol,
        "records": records,
        "statistics": statistics,
    }
    atomic_json_dump(payload, output_json)
    write_csv(records, output_csv)
    write_npz(records, modes, output_npz)
    partial_json.unlink(missing_ok=True)

    print("\n" + "#" * 112)
    print("EXPORT COMPLETE")
    print("#" * 112)
    print(output_json)
    print(output_csv)
    print(output_npz)
    if args.quick:
        print("WARNING: --quick output is a pipeline audit, not a paper result.")


if __name__ == "__main__":
    main()
