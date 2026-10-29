"""Bounded bidirectional joint-space transit planner; no contact or waist projection."""
import time
import numpy as np


def plan_transit(start,goal,lower,upper,indices,check,seconds=8.,iterations=600):
    start,goal=np.asarray(start,float),np.asarray(goal,float)
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    indices=list(indices);rng=np.random.default_rng(19);deadline=time.monotonic()+seconds
    check(start);check(goal)
    def edge(a,b):
        count=max(1,int(np.ceil(np.max(abs(b-a)/np.array([.001]+[.01]*15)))))
        for t in np.linspace(0,1,count+1):
            if time.monotonic()>deadline:raise TimeoutError('碰撞绕行规划超时')
            try:check(a*(1-t)+b*t)
            except ValueError:return False
        return True
    trees=[([start.copy()],[-1]),([goal.copy()],[-1])]
    def extend(tree,target):
        nodes,parents=tree
        nearest=int(np.argmin([np.linalg.norm((q-target)[indices]) for q in nodes]))
        delta=target-nodes[nearest];length=float(np.linalg.norm(delta[indices]))
        q=nodes[nearest]+delta*min(1.,.12/max(length,1e-12))
        if not edge(nodes[nearest],q):return None
        nodes.append(q);parents.append(nearest);return len(nodes)-1
    def branch(tree,index):
        result=[]
        while index>=0:result.append(tree[0][index]);index=tree[1][index]
        return list(reversed(result))
    for iteration in range(iterations):
        if time.monotonic()>deadline:break
        which=iteration%2;a,b=trees[which],trees[1-which]
        target=b[0][0].copy() if iteration%5==0 else start.copy()
        if iteration%5:target[indices]=rng.uniform(lower[indices],upper[indices])
        ia=extend(a,target)
        if ia is None:continue
        for _ in range(100):
            ib=extend(b,a[0][ia])
            if ib is None:break
            if np.linalg.norm((b[0][ib]-a[0][ia])[indices])<1e-8:
                left,right=branch(a,ia),branch(b,ib)
                path=left+list(reversed(right[:-1]))
                if which:path=list(reversed(path))
                # All returned adjacent knots obey the original jump limits.
                rows=[path[0]]
                for x,y in zip(path,path[1:]):
                    steps=max(1,int(np.ceil(np.max(abs(y-x))/.03)))
                    rows.extend(np.linspace(x,y,steps+1)[1:])
                return np.asarray(rows).tolist()
    raise ValueError('碰撞绕行未找到路径；未证明目标不可达')
