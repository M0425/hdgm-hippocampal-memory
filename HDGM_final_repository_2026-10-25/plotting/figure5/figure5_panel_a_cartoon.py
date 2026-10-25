"""Vector cue cartoons for a conceptual perturbation diagram.

Houses and trees are illustrative cue symbols, not depictions of the actual
experimental stimulus set. No measured values are encoded by these icons.
"""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle, FancyArrowPatch


def cue(ax, x, y, kind='house', noisy=False, faded=False):
    """Small rounded sensory-cue card, drawn entirely with vector shapes."""
    ax.add_patch(FancyBboxPatch((x-.051,y-.071),.102,.142,
        boxstyle='round,pad=.006,rounding_size=.012',fc='white',ec='#C3CCD3',lw=.8,zorder=2))
    ax.plot([x-.041,x+.041],[y-.050,y-.050],color='#D7DFDB',lw=.7,zorder=3)
    if kind=='house':
        ax.add_patch(Rectangle((x-.031,y-.045),.062,.064,
            fc='#F1DBD3',ec='#967B76',lw=.65,zorder=3))
        ax.add_patch(Polygon([(x-.042,y+.013),(x,y+.052),(x+.042,y+.013)],
            fc='#C96E79' if not faded else '#A98D9D',ec='#967B76',lw=.7,zorder=4))
        ax.add_patch(Rectangle((x-.009,y-.045),.018,.032,fc='#9DAFC0',ec='none',zorder=4))
        ax.add_patch(Rectangle((x+.014,y-.012),.010,.017,fc='#FFF8E7',ec='#B99C90',lw=.45,zorder=4))
    else:
        ax.add_patch(Rectangle((x-.007,y-.047),.014,.072,fc='#B49C88',ec='none',zorder=3))
        for dx,dy,r in [(-.020,.004,.022),(.018,.009,.024),(0,.033,.026)]:
            ax.add_patch(Circle((x+dx,y+dy),r,fc='#8CA995',ec='#607E6A',lw=.6,zorder=4))
    if noisy:
        rng=np.random.default_rng(54)
        ax.scatter(x+rng.uniform(-.045,.045,34),y+rng.uniform(-.062,.061,34),
            s=rng.uniform(2,6,34),c='#74869A',alpha=.47,linewidths=0,zorder=6)


def draw(ax, old):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    for x,fill,edge in [(0.015,old.ALIAS_LIGHT,old.ALIAS),(.515,old.NOISE_LIGHT,old.NOISE)]:
        ax.add_patch(FancyBboxPatch((x,.255),.47,.69,
            boxstyle='round,pad=.01,rounding_size=.022',
            fc=old.blend(fill,.53),ec=old.blend(edge,.32),lw=.8))
    ax.text(.043,.901,'STRUCTURAL ALIASING',color=old.ALIAS_DARK,fontsize=9,weight='bold')
    ax.text(.543,.901,'OBSERVATION NOISE',color=old.NOISE_DARK,fontsize=9,weight='bold')
    ax.text(.043,.847,'Distinct inputs converge',color=old.INK,fontsize=9,weight='bold')
    ax.text(.543,.847,'Structure survives perturbation',color=old.INK,fontsize=9,weight='bold')
    cue(ax,.087,.711,'house');cue(ax,.205,.711,'tree')
    cue(ax,.388,.711,'house',faded=True)
    cue(ax,.612,.711,'house');cue(ax,.886,.711,'house',noisy=True)
    for start,end in [((.268,.711),(.323,.711)),((.688,.711),(.815,.711))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,
            lw=1.1,color=old.SLATE,zorder=6))
    ax.text(.388,.603,'aliased cue',ha='center',fontsize=8,color=old.ALIAS_DARK)
    ax.text(.146,.603,'distinct cues',ha='center',fontsize=8,color=old.SLATE)
    ax.text(.612,.603,'clean cue',ha='center',fontsize=8,color=old.SLATE)
    ax.text(.886,.603,'noisy cue',ha='center',fontsize=8,color=old.NOISE_DARK)
    v=np.array([.95,.15,.8,.2,.65,.25,.9,.1,.75,.35])
    vn=np.array([.85,.2,.7,.32,.95,.18,.78,.25,.62,.35])
    old.draw_population(ax,.063,.497,v,old.ALIAS,.74)
    old.draw_population(ax,.063,.405,v[::-1],old.HISTORY,.74)
    old.draw_population(ax,.331,.450,.53*v+.47*v[::-1],old.ALIAS,.80)
    old.draw_population(ax,.560,.450,vn,old.NOISE,.80)
    old.draw_population(ax,.827,.450,vn,old.NOISE,.80,jitter=True)
    for start,end in [((.228,.450),(.312,.450)),((.701,.450),(.798,.450))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,
            color=old.SLATE,lw=1))
    ax.text(.25,.313,r'$\alpha\uparrow$ erases discriminative structure',ha='center',fontsize=8,color=old.SLATE)
    ax.text(.75,.313,r'$\sigma\uparrow$ adds stochastic variation',ha='center',fontsize=8,color=old.SLATE)
    # Retain the original conceptual EC–CA3–CA1 route below the cartoons.
    ax.text(.03,.145,'same frozen\ncheckpoint',va='center',fontsize=8,color=old.SLATE)
    for x,name,color in [(.33,'EC',old.HISTORY),(.51,'CA3',old.DECISION),(.69,'CA1',old.NOISE)]:
        ax.add_patch(Circle((x,.145),.044,fc='white',ec=color,lw=1.25))
        ax.text(x,.145,name,ha='center',va='center',fontsize=8,color=color,weight='bold')
    for xa,xb in [(.378,.462),(.558,.642),(.738,.806)]:
        ax.add_patch(FancyArrowPatch((xa,.145),(xb,.145),arrowstyle='-|>',
            mutation_scale=9,color=old.MID,lw=.9))
    ax.text(.825,.145,'choice',va='center',fontsize=8,color=old.INK)
    ax.text(.5,.046,'One factor varied at a time · cue cartoons are illustrative',
        ha='center',fontsize=8,color=old.SLATE)
