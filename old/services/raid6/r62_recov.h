/* This work is licensed under the terms of the GNU GPL, version 2.  See
 * the GPL-COPYING file in the top-level directory.
 *
 * Copyright (c) 2010-2011 University of Utah and the Flux Group.
 * All rights reserved.
 */

#ifndef __R62_RECOV_H__
#define __R62_RECOV_H__

struct r62_tbl {
    int pbidx;
    int qidx;
};

struct r62_recov_data {
    size_t bytes;
    int n;
    struct r62_tbl idx[0];
};

#endif
