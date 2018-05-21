import t4k
from safe import safe_min, safe_lte
from grouper import trim

def equals(a,b):
    return a==b


class Aligner(object):

    LOWER = 0
    UPPER = 1

    def __init__(self, shortcut=True):
        self.queue = None
        self.seen = None
        self._path = None
        self.shortcut = shortcut
        pass


    def align(self, s1, s2):
        self.s1, self.s2, self.l1, self.l2 = s1, s2, len(s1), len(s2)
        self.queue = ([],[])
        self.seen = {}
        self._path = {}
        i, j, best = 0, 0, abs(self.l1 - self.l2)
        if self.enqueue_lower(i, j, best):
            return 0
        return self.process_queue()


    def path(self):
        path = []
        i, j = self.l1, self.l2
        while i > 0 and j > 0:
            c = self._path[i,j]
            path.append(c)
            if c == '\\' or c == '`':
                i, j = i-1, j-1
            elif c == '=' or c == '>':
                i, j = i, j-1
            elif c == '|' or c == 'v':
                i, j = i-1, j

        path.reverse()
        return path


    def process_queue(self):

        while True:

            # Get the next item.  Potentially skip it.  Check termination.
            i, j, best, skip, c = self.pop()
            if skip: continue
            self._path[i,j] = c
            if i == self.l1 and j == self.l2: return best

            # Enqueue match / substitution
            diag_match = False
            enqueue_diag = None
            if self.in_bounds(i+1,j+1):
                diag_match = self.s1[i] == self.s2[j]
                if diag_match:
                    enqueue_diag = (self.LOWER, i+1, j+1, best, '\\')
                else:
                    enqueue_diag = (self.UPPER, i+1, j+1, best+2, '`')

            # Enqueue insertion
            enqueue_right = None
            if self.in_bounds(i, j+1):
                if self.l1 - i < self.l2 - j:
                    if not self.shortcut or c == '\\' or c == '=':
                        enqueue_right = (self.LOWER, i, j+1, best, '=')
                elif not diag_match:
                    enqueue_right= (self.UPPER, i, j+1, best+2, '>')

            # Enqueue deletion
            enqueue_down = None
            if self.in_bounds(i+1, j):
                if self.l2 - j < self.l1 - i:
                    if not self.shortcut or c == '\\' or c == '|':
                        enqueue_down = (self.LOWER, i+1, j, best, '|')

                elif not diag_match: 
                    enqueue_down = (self.UPPER, i+1, j, best+2, 'v')

            # Do the enqueuing, but enqueue diag last, so it if first out later.
            if enqueue_down is not None:  self.enqueue(*enqueue_down)
            if enqueue_right is not None: self.enqueue(*enqueue_right)
            if enqueue_diag is not None:  self.enqueue(*enqueue_diag)


    def print_seen(self):

        # First collect the content we want to print.
        rows = []
        for i in range(self.l1 + 1):
            row = []
            rows.append(row)
            for j in range(self.l2 + 1):
                try:
                    row.append(self._path[i,j])
                except KeyError:
                    row.append('_')

        # Now print it.
        padded_s1 = self.s1 + ' '
        print '   ' + self.s2
        for i, row in enumerate(rows):
            print padded_s1[i-1] + ' ' + ''.join(row)


    def enqueue(self, which, i, j, best, c='*'):
        if which == self.LOWER:
            self.enqueue_lower(i, j, best, c)
        elif which == self.UPPER:
            self.enqueue_upper(i, j, best, c)
        else:
            raise ValueError('Which must be aligner.LOWER or aligner.UPPER.')


    def enqueue_lower(self, i, j, best, c='*'):
        """
        Enqueue another branch to the lower queue.  Check if this item has
        been enqueued before.  If so, only enqueue it if we have a new best
        for that item (and in that case, flag the old entry to be skipped).
        """

        # If this hasn't been enqueued before, add it to the queue
        if not (i,j) in self.seen:
            self.queue[0].append((i, j, best, False, c))
            location = len(self.queue[0])
            self.seen[i,j] = (location, best)

        # If a prior entry exists, only re-enqueue if this one is better
        old_location, old_best = self.seen[i,j]
        if best < old_best:

            # Flag the old entry to be skipped (if it still exists)
            try:
                old_entry_exists = self.queue[1][old_location][:2] == (i,j)
            except IndexError:
                old_entry_exists = False

            if old_entry_exists:
                old_entry = self.queue[1][old_location]
                marked_to_skip = old_entry[:-2] + (True,) + old_entry[-1:]
                self.queue[1][old_location] = marked_to_skip

            # In either case, enqueue the new value and update seen
            self.queue[0].append((i,j,best,False, c))
            location = len(self.queue[0])
            self.seen[i,j] = (location, best)


    def enqueue_upper(self, i, j, best, c='*'):

        # If this has been enqueued before, then certainly this one can't be
        # better because it is being enqueued on the upper queue.
        if (i,j) in self.seen: return

        # Otherwise add it to the queue as new
        self.queue[1].append((i, j, best, False, c))
        location = len(self.queue[0])
        self.seen[i,j] = (location, best)

    
    def in_bounds(self, i, j):
        if i <= self.l1 and j <= self.l2:
            return True
        return False


    def pop(self):
        try:
            return self.queue[0].pop()
        except IndexError:
            self.queue = (self.queue[1],[])
            return self.queue[0].pop()



#class FrameStack(list):
#
#    def __init__(self):
#        super(FrameStack, self).__init__()
#        self.append({'state': 'init'})
#
#    def call(self, func, args=None):
#        args = args or {}
#        caller = self.cur_frame()
#        caller['state'] = 'calling_' + func
#        callee = {
#            'state': 'starting',
#            'func': func,
#            'locals': dict(args)
#        }
#        self.append(callee)
#
#    def reply(self, return_val):
#        callee_frame = self.pop()
#        self.cur_frame()['state'] = 'return_' + callee_frame['func']
#        self.cur_frame()['return'] = return_val
#
#    def locals(self):
#        return self.cur_fame()['locals']
#
#    def state(self):
#        return self.cur_frame()['state']
#
#    def func(self):
#        return self.cur_frame()['func']
#
#    def return_val(self):
#        return self.cur_frame()['return']
#
#    def cur_frame(self):
#        return self[-1]
#
#    def empty(self):
#        return len(self) == 1
#
#    def __getitem__(self, key):
#        if isinstance(key, int):
#            return super(FrameStack, self).__getitem__(key)
#        return self.cur_frame()['locals'][key]
#
#    def __setitem__(self, key, val):
#        self.cur_frame()['locals'][key] = val
#
#
#class FastAlignBadState(Exception):
#    pass
#
#
#class AlignerNoRecurse(object):
#
#    DOWN = 0
#    RIGHT = 1
#    
#    def __init__(self):
#        self.distance = None
#        self.path = None
#
#
#    def align(self, seq1, seq2, equals=equals):
#
#        self.seq1, self.seq2 = seq1, seq2
#        self.len1, self.len2 = len(seq1), len(seq2)
#        self.equals = equals
#
#        finished = False
#        i, j, penalty, best = 0, 0, 0, 0
#        fs = FrameStack()
#        finished = False
#        while not finished:
#            fs.call('calculate', {
#                'i':i,
#                'j':j,
#                'penalty':penalty,
#                'best':best,
#                'depth':0
#            })
#            finished, best = self.process_frames(fs)
#
#        return finished, best
#
#
#    def process_frames(self, fs):
#        while not fs.empty():
#            print (
#                ' ' * fs['depth'] + fs.func() + ' ' + fs.state() 
#                + (' i:%d'%fs['i']) + (' j:%d'%fs['j']) 
#                + (' p:%d'%fs['penalty']) + (' b:%d'%fs['best'])
#            )
#            getattr(self, fs.func())(fs)
#        return fs.return_val()
#
#
#    def calculate(self, fs):
#
#        if fs.state() == 'starting':
#
#            # Check if this path's `new_best` is already too large.
#            fs['new_best'] = bound_best(fs['i'], fs['j'], fs['penalty'])
#            if fs['new_best'] > fs['best']:
#                fs.reply((False, fs['new_best']))
#                return
#
#            # Try moving diagonal first.
#            fs.call('maybe_calc_diag', {
#                'i': fs['i']+1,
#                'j': fs['j']+1,
#                'penalty': fs['penalty'],
#                'best': fs['best'],
#                'depth': fs['depth']+1
#            })
#            return
#
#        if fs.state() == 'return_maybe_calc_diag':
#
#            # Collect results from calling maybe_calc_diag.
#            fs['diag_finished'], fs['diag_best'] = fs.return_val()
#            if fs['diag_finished']:
#                fs.reply((True, fs['diag_best']))
#                return
#
#            # Try going right
#            fs['went'] = 'right'
#            fs.call('maybe_calc_rd', {
#                'i': fs['i'],
#                'j': fs['j']+1,
#                'penalty': fs['penalty'],
#                'best': fs['best'],
#                'depth': fs['depth']+1
#            })
#            return
#
#        if fs.state() == 'return_maybe_calc_rd' and fs['went'] == 'right':
#
#            # Collect results from calling maybe_calc_rd and going right
#            fs['right_finished'], fs['right_best'] = fs.return_val()
#            if fs['right_finished']:
#                fs.reply((True, fs['right_best']))
#                return
#
#            # Try going down
#            fs['went'] = 'down'
#            fs.call('maybe_calc_rd', {
#                'i': fs['i']+1,
#                'j': fs['j'],
#                'penalty': fs['penalty'],
#                'best': fs['best'],
#                'depth': fs['depth']+1
#            })
#            return
#
#        if fs.state() == 'return_maybe_calc_rd' and fs['went'] == 'down':
#
#            # Collect results from calling maybe_calc_rd and going down
#            fs['down_finished'], fs['down_best'] = fs.return_val()
#            if fs['down_finished']:
#                fs.reply((True, fs['down_best']))
#                return
#
#            # If all the childrens' best is None, it means we're in the
#            # bottom corner, i.e. we're done.
#            fs['children_best'] = t4k.safe_min(
#                fs['diag_best'], fs['right_best'], fs['down_best'])
#            if fs['children_best'] is None:
#                fs.reply((True, fs['penalty']))
#                return
#
#            # Otherwise propogate up the best penalty found by searching
#            # along this branch.
#            fs.reply((False, fs['children_best']))
#            return
#
#        raise FastAlignBadState('Bad State: %s' % str(fs))
#
#
#    def maybe_calc_diag(self, fs):
#
#        if fs.state() == 'starting':
#
#            if not self.in_bounds(fs['i'], fs['j']):
#                fs.reply((False, None))
#                return
#
#            # If these items are equal, make a recursive call to `calculate`
#            # without adding anything to the penalty
#            a, b = self.seq1[fs['i']-1], self.seq2[fs['j']-1]
#            if self.equals(a, b):
#                fs.call('calculate', {
#                    'i': fs['i'],
#                    'j': fs['j'],
#                    'penalty': fs['penalty'],
#                    'best': fs['best'],
#                    'depth': fs['depth']
#                })
#                return
#
#            # Otherwise, make the recursive call and add 2 to the penalty.
#            fs.call('calculate', {
#                'i': fs['i'],
#                'j': fs['j'],
#                'penalty': fs['penalty'] + 2,
#                'best': fs['best'],
#                'depth': fs['depth']
#            })
#            return
#
#        # Return what we got from the recursive call to `calculate`.
#        if fs.state() == 'return_calculate':
#            finished, best = fs.return_val()
#            fs.reply((finished, best))
#            return
#
#        raise FastAlignBadState('Bad State: %s' % str(fs))
#
#
#    def maybe_calc_rd(self, fs):
#
#        if fs.state() == 'starting':
#            if not self.in_bounds(fs['i'], fs['j']):
#                fs.reply((False, None))
#                return
#
#            fs.call('calculate', {
#                'i': fs['i'],
#                'j': fs['j'],
#                'penalty': fs['penalty']+1,
#                'best': fs['best'],
#                'depth': fs['depth']
#            })
#            return
#
#        if fs.state() == 'return_calculate':
#            finished, best = fs.return_val()
#            fs.reply((finished, best))
#            return
#
#        raise FastAlignBadState('Bad State: %s' % str(fs))
#
#
#    def in_bounds(self, i, j):
#        if i > self.len1:
#            return False
#        if j > self.len2:
#            return False
#        return True
#
#
#
#class AlignerRecurse(object):
#
#    DOWN = 0
#    RIGHT = 1
#    
#    def __init__(self):
#        self.distance = None
#        self.path = None
#
#
#    def align(self, seq1, seq2, equals=equals):
#
#        self.seq1, self.seq2 = seq1, seq2
#        self.len1, self.len2 = len(seq1), len(seq2)
#        self.equals = equals
#        finished = False
#        i, j, penalty, best = 0, 0, 0, 0
#        while not finished:
#            print '\nstart a dive\n' + '='*30
#            finished, best = self.calculate(0, 0, penalty, best)
#
#        return finished, best
#
#
#    def maybe_calc_diag(self, i, j, penalty, best, depth):
#        print ' ' * depth + 'in bounds? %s' % str(self.in_bounds(i,j))
#        if not self.in_bounds(i, j):
#            return False, None
#        a, b = self.seq1[i-1], self.seq2[j-1]
#        is_equal = self.equals(a, b)
#        print ' ' * depth + 'equal? %s %s %s' % (a, b, str(is_equal))
#        had_match = False
#        if self.equals(self.seq1[i-1], self.seq2[j-1]):
#            had_match = True
#            finished, best = self.calculate(i, j, penalty, best, depth)
#            return had_match, finished, best
#        finished, best = self.calculate(i, j, penalty+2, best, depth)
#        return had_match, finished, best
#
#
#    def maybe_calc_rd(self, i, j, penalty, best, depth):
#        if not self.in_bounds(i, j):
#            return False, None
#        return self.calculate(i, j, penalty+1, best, depth)
#
#
#    def calculate(self, i, j, penalty=0, best=0, depth=0):
#
#        print ' ' * depth + 'starting %d, %d, %d, %d' % (i, j, penalty, best)
#        # Check if this path's `new_best` is already too large.
#        new_best = bound_best(i, j, penalty)
#        print ' ' * depth + 'test new best: %d, %d' % (best, new_best)
#        if new_best > best:
#            print ' ' * depth + 'back up!'
#            return 0, False, new_best
#
#        # Try moving diagonal first.
#        print ' ' * depth + 'trying diag...'
#        had_match, diag_finished, diag_best = self.maybe_calc_diag(
#            i+1, j+1, penalty, best, depth+1)
#        print (
#            ' ' * depth + 'got diag: %s, %d, %s' 
#            % (had_match, best, str(diag_best))
#        )
#        if diag_finished:
#            return diag_finished, diag_best
#
#        # Now try going right.
#        print ' ' * depth + 'trying right...'
#        right_finished, right_best = self.maybe_calc_rd(
#            i, j+1, penalty, best, depth+1)
#        print ' ' * depth + 'got right: %d, %s' % (best, str(right_best))
#        if right_finished:
#            return right_finished, right_best
#
#        # Finally try going down.
#        print ' ' * depth + 'trying down...'
#        down_finished, down_best = self.maybe_calc_rd(
#            i+1, j, penalty, best, depth+1)
#        print ' ' * depth + 'got down: %d, %s' % (best, str(down_best))
#        if down_finished:
#            return down_finished, down_best
#
#        # If all the childrens' best is None, it means we're in the bottom 
#        # corner, i.e. we're done
#        children_best = t4k.safe_min(diag_best, right_best, down_best)
#        print (
#            ' ' * depth + 'check children: %d, %s' % (best, str(children_best)))
#        if children_best is None:
#            print (
#                ' ' * depth + 'return: %s, %d, %s' 
#                % (True, best, str(children_best))
#            )
#            return True, penalty
#
#        # Otherwise propogate up the best penalty found by searching along this
#        # branch.
#        print (
#            ' ' * depth + 'return: %s, %d, %s' 
#            % (False, best, str(children_best))
#        )
#        return False, children_best
#
#
#    def in_bounds(self, i, j):
#        if i > self.len1:
#            return False
#        if j > self.len2:
#            return False
#        return True
#
#
## This assumes equal-length strings
#def bound_best(i, j, penalty):
#    return penalty + abs(i-j)


