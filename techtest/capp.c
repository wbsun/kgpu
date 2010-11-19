#include "build/include/linux/module.h"
#include "build/include/linux/kernel.h"
#include "build/include/linux/init.h"

#include "build/include/linux/kthread.h"
#include "build/include/linux/netfilter.h"
#include "build/include/linux/ip.h"
#include "build/include/linux/netfilter_ipv4.h"
#include "build/include/linux/netdevice.h"

#include "build/include/linux/proc_fs.h"
#include "build/arch/x86/include/asm/uaccess.h"
#include "build/arch/x86/include/asm/page.h"
#include "build/include/linux/mm.h"
#include "build/include/linux/mm_types.h"
#include "build/include/linux/types.h"

#define PROCFS_NAME "capp"

void *__uvirt_to_phys(void *addr)
{
    unsigned long kva, ret;
    //unsigned long flags;
    struct page *pg;
    //struct vm_area_struct *vma;

    spin_lock(&current->mm->page_table_lock);

    
    if (get_user_pages(current, current->mm,
		       (unsigned long)addr, 1, 1, 0,
		       &pg, NULL)<0) {
	spin_unlock(&current->mm->page_table_lock);
	printk(KERN_ALERT "[capp] Error: get user page fault\n");

	return 0;
    }

    kva = (unsigned long)page_address(pg);
    kva |= (unsigned long)addr & (PAGE_SIZE-1); /* restore the offset */

    ret = __pa(kva);

    spin_unlock(&current->mm->page_table_lock);

    return (void *)ret;
}        

union{
    char *buf;
    struct {
	void *startaddr;
        unsigned long size;
    } info;
} capp_comm_mem[2];

static volatile int *psa[2];
	
static struct proc_dir_entry *capp_procfile;

static struct task_struct *cappkts;
static struct nf_hook_ops cappops[5];

static int __capp_procfs_read(char *buf, char **bufloc,
			      off_t offset, int buflen,
			      int *eof, void *data)
{
    int ret;

    if (offset>0)
	ret = 0;
    else {
	memcpy(buf, (char*)capp_comm_mem, sizeof(capp_comm_mem));
	ret = sizeof(capp_comm_mem);
    }

    return ret;
}

static int __capp_procfs_write(struct file *file, const char *buf,
			       unsigned long count, void *data)
{
    if (count < sizeof(capp_comm_mem))
	return -EFAULT;
    if ( copy_from_user((char*)capp_comm_mem, buf, sizeof(capp_comm_mem)))
	return -EFAULT;

    psa[0] = __uvirt_to_phys(capp_comm_mem[0].info.startaddr);
    psa[1] = __uvirt_to_phys(capp_comm_mem[1].info.startaddr);
    if (psa[0] == NULL){
	capp_comm_mem[0].info.startaddr = NULL;
	printk(KERN_INFO "[capp] ERROR: can't get phy addr of %lu\n",
	       (unsigned long)(capp_comm_mem[0].info.startaddr));
    }else{
	*((unsigned int*)phys_to_virt((phys_addr_t)psa[0])) = 1;
    }
    
    
    return sizeof(capp_comm_mem);
}

static void __capp_register_procfs(void)
{
    capp_comm_mem[0].info.startaddr = NULL;
    capp_comm_mem[1].info.startaddr = NULL;
    capp_comm_mem[0].info.size = 0;
    capp_comm_mem[1].info.size = 0;
    psa[0] = NULL;
    psa[1] = NULL;
    
    capp_procfile = create_proc_entry(PROCFS_NAME, 0777, NULL);

    if (capp_procfile == NULL) {
	remove_proc_entry(PROCFS_NAME, NULL);
	printk(KERN_ALERT "[capp] Error: Could not initialize /proc/%s\n",
	       PROCFS_NAME);
	return;
    }

    capp_procfile->read_proc = __capp_procfs_read;
    capp_procfile->write_proc = __capp_procfs_write;
    //capp_procfile->owner = THIS_MODULE;
    capp_procfile->mode = S_IFREG|S_IRUGO;
    capp_procfile->uid = 0;
    capp_procfile->gid = 0;
    capp_procfile->size = sizeof(capp_comm_mem);
}


static unsigned int __capp_hook(unsigned int hooknum, struct sk_buff *skb,
		     const struct net_device *in,
		     const struct net_device *out,
		     int (*okfn)(struct sk_buff *))
{
    struct iphdr *iph;
    static unsigned long pnum = 0;
    
    if (!skb)
	return NF_ACCEPT;

    iph = (struct iphdr *)skb_network_header(skb);
    if (!iph)
	return NF_ACCEPT;

    pnum++;

    //return NF_ACCEPT;

    switch(hooknum){
    case NF_INET_PRE_ROUTING:
	printk(KERN_INFO "[CAPP]: pre %lu, %s %d.%d.%d.%d => %s %d.%d.%d.%d\n",
	       pnum, in?in->name:"", iph->saddr&0xff, (iph->saddr&0xff00)>>8,
	   (iph->saddr&0xff0000)>>16, (iph->saddr&0xff000000)>>24,
	       out?out->name:"", iph->daddr&0xff, (iph->daddr&0xff00)>>8,
	   (iph->daddr&0xff0000)>>16, (iph->daddr&0xff000000)>>24);
	
	if ((iph->daddr&0xff) == 10 && (iph->daddr&0xff00)>>8 == 0 &&
	    (iph->daddr&0xff0000)>>16 == 4 && (iph->daddr&0xff000000)>>24 == 1)
	    iph->daddr = (iph->daddr&0x00ffffff)|0x2000000;
	
	break;
    case NF_INET_FORWARD:
	printk(KERN_INFO "[CAPP]: fwd %lu, %s %d.%d.%d.%d => %s %d.%d.%d.%d\n",
	       pnum, in?in->name:"", iph->saddr&0xff, (iph->saddr&0xff00)>>8,
	   (iph->saddr&0xff0000)>>16, (iph->saddr&0xff000000)>>24,
	       out?out->name:"", iph->daddr&0xff, (iph->daddr&0xff00)>>8,
	   (iph->daddr&0xff0000)>>16, (iph->daddr&0xff000000)>>24);
	break;
    case NF_INET_POST_ROUTING:
	printk(KERN_INFO "[CAPP]: post %lu, %s %d.%d.%d.%d => %s %d.%d.%d.%d\n",
	       pnum, in?in->name:"", iph->saddr&0xff, (iph->saddr&0xff00)>>8,
	   (iph->saddr&0xff0000)>>16, (iph->saddr&0xff000000)>>24,
	       out?out->name:"", iph->daddr&0xff, (iph->daddr&0xff00)>>8,
	   (iph->daddr&0xff0000)>>16, (iph->daddr&0xff000000)>>24);
	break;
    case NF_INET_LOCAL_IN:
	printk(KERN_INFO "[CAPP]: in %lu, %s %d.%d.%d.%d => %s %d.%d.%d.%d\n",
	       pnum, in?in->name:"", iph->saddr&0xff, (iph->saddr&0xff00)>>8,
	   (iph->saddr&0xff0000)>>16, (iph->saddr&0xff000000)>>24,
	       out?out->name:"", iph->daddr&0xff, (iph->daddr&0xff00)>>8,
	   (iph->daddr&0xff0000)>>16, (iph->daddr&0xff000000)>>24);
	break;
    case NF_INET_LOCAL_OUT:
	printk(KERN_INFO "[CAPP]: out %lu, %s %d.%d.%d.%d => %s %d.%d.%d.%d\n",
	       pnum, in?in->name:"", iph->saddr&0xff, (iph->saddr&0xff00)>>8,
	   (iph->saddr&0xff0000)>>16, (iph->saddr&0xff000000)>>24,
	       out?out->name:"", iph->daddr&0xff, (iph->daddr&0xff00)>>8,
	   (iph->daddr&0xff0000)>>16, (iph->daddr&0xff000000)>>24);
	break;
    default:
	break;
    }
    
    return NF_ACCEPT; /*NF_QUEUE;*/
}

static void __init_hookops(void)
{
    cappops[0].hook = __capp_hook;
    cappops[0].hooknum = NF_INET_PRE_ROUTING; 
    cappops[0].pf = PF_INET;
    cappops[0].priority = NF_IP_PRI_FIRST;

    cappops[1].hook = __capp_hook;
    cappops[1].hooknum = NF_INET_FORWARD;; 
    cappops[1].pf = PF_INET; /*NFPROTO_IPV4;*/
    cappops[1].priority = NF_IP_PRI_FIRST;

    cappops[2].hook = __capp_hook;
    cappops[2].hooknum = NF_INET_POST_ROUTING; 
    cappops[2].pf = PF_INET; /*NFPROTO_IPV4;*/
    cappops[2].priority = NF_IP_PRI_FIRST;

    cappops[3].hook = __capp_hook;
    cappops[3].hooknum = NF_INET_LOCAL_IN;; 
    cappops[3].pf = PF_INET; /*NFPROTO_IPV4;*/
    cappops[3].priority = NF_IP_PRI_FIRST;

    cappops[4].hook = __capp_hook;
    cappops[4].hooknum = NF_INET_LOCAL_OUT; 
    cappops[4].pf = PF_INET; /*NFPROTO_IPV4;*/
    cappops[4].priority = NF_IP_PRI_FIRST;

}

static int __capp_kthread(void *d)
{
    static unsigned int num = 0;
    
    while(1) {
	msleep(100);
	if (kthread_should_stop())
	    break;
	if (capp_comm_mem[0].info.startaddr != NULL) {
	    *((unsigned int*)phys_to_virt((phys_addr_t)psa[0])) = num;
	}else{
	    printk(KERN_INFO "addr is 0\n");
	}
	num++;
    }

    return 0;
}

static int __init capp_init(void)
{
    __init_hookops();
    __capp_register_procfs();
    printk(KERN_INFO "Load capp module and start the capp kthread\n");
    cappkts = kthread_run(__capp_kthread, NULL, "capp_kthread");
    nf_register_hooks(cappops, 5);
    
    return 0;
}

static void __exit capp_exit(void)
{
    printk(KERN_INFO "Stop capp kthread and unload capp module\n");
    kthread_stop(cappkts);
    nf_unregister_hooks(cappops, 5);
    remove_proc_entry(PROCFS_NAME, NULL);
}

module_init(capp_init);
module_exit(capp_exit);
