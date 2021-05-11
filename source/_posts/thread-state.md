---
title: Thread.State
catalog: true
date: 2019-04-25 19:57:24
subtitle: 线程之间的状态
header-img:
tags: 多线程
---

# 线程直接的状态主要有如下几种
- NEW 就绪
- RUNNABLE 线程运行中或I/O等待
- BLOCKED 线程在等待monitor锁(synchronized关键字)
- WAITING 线程在无限等待唤醒
- TIME_WAITING 线程在等待唤醒，但设置了时限


# 几种线程关键字的解释
sleep: 线程进入等待状态，获得到了cpu进行等待时间，即等待状态
yield: 线程释放cpu，重新进入就绪状态，与其他线程一同抢占cpu
join: thread.join()，即thread插入当前线程之前，在运行完thread之后，当前线程继续执行
wait: 是Object下的方法，wait()的作用是让当前线程进入等待状态，同时，wait()也会让当前线程释放它所持有的锁
notify, notifyAll: notify()和notifyAll()的作用，则是唤醒当前对象上的等待线程；notify()是唤醒单个线程，而notifyAll()是唤醒所有的线程。