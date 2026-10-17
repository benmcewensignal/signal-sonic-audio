"""Candidate fingerprinters, for the room test to compare with the one in use (audio.fingerprints).

The room test of 17 October found that a 0.6-second reverb leaves 2% of the fingerprints in place: the tail fills the
spectrogram and changes which points are local maxima. The pairing and the hash are the same as audio.fingerprints; only
the choice of peaks differs:
  base     the fingerprinter in use (audio.fingerprints._peaks)
  wide     a longer local-maximum window in time (0.56 s rather than 0.28), so a tail's spurious peaks lose to the onset
  onset    peaks of the rise: the spectrogram minus its running maximum over the previous 0.28 s, so decays do not count
  whiten   peaks of the local contrast: each bin minus its own average over the surrounding 1.5 s, so sustained energy
           (a reverb tail, a pad) does not count and only what stands out of its own recent past does
  onsetwide  onset peaks with the longer window
Every candidate is a drop-in: hashes(y, name) returns [(hash32, anchor_frame)] like audio.fingerprints.hashes.
"""
import numpy as np
from . import fingerprints as FP

NAMES = ["base", "wide", "onset", "whiten", "onsetwide"]


def _sdb(y):
    S = np.abs(np.fft.rfft(FP._frames(y), axis=1))
    Sdb = 20 * np.log10(S + 1e-9)
    return Sdb - Sdb.max()


def _pick(A, wt, wf, floor_db, Sdb=None, rise_db=None):
    """Local maxima of A within (wt frames, wf bins), at least floor_db on the raw spectrogram Sdb (A itself when None),
    and, for a difference spectrogram, at least rise_db above zero."""
    neigh = FP._local_max_filter(A, wt, wf)
    mask = (A >= neigh - 1e-9)
    mask &= ((Sdb if Sdb is not None else A) >= floor_db)
    if rise_db is not None: mask &= (A >= rise_db)
    mask[:, :2] = False; mask[:, -2:] = False
    ts, fs = np.nonzero(mask)
    order = np.argsort(ts, kind="stable")
    return list(zip(ts[order].tolist(), fs[order].tolist()))


def _past_max(Sdb, k):
    """Each cell's maximum over the k frames before it (not including itself)."""
    out = np.full_like(Sdb, -200.0)
    for d in range(1, k + 1):
        out[d:] = np.maximum(out[d:], Sdb[:-d])
    return out


def _local_mean(Sdb, k):
    """Each bin's mean over the surrounding 2k+1 frames."""
    c = np.cumsum(np.pad(Sdb, ((k + 1, k), (0, 0)), mode="edge"), axis=0)
    return (c[2 * k + 1:] - c[:-(2 * k + 1)]) / (2 * k + 1)


def peaks(y, name):
    if name == "base":
        return FP._peaks(y)
    Sdb = _sdb(y)
    if name == "wide":
        return _pick(Sdb, 24, FP.PEAK_NEIGH_F, FP.MIN_PEAK_DB)
    if name == "onset":
        D = Sdb - _past_max(Sdb, FP.PEAK_NEIGH_T)
        return _pick(D, FP.PEAK_NEIGH_T, FP.PEAK_NEIGH_F, FP.MIN_PEAK_DB, Sdb=Sdb, rise_db=3.0)
    if name == "onsetwide":
        D = Sdb - _past_max(Sdb, 24)
        return _pick(D, 24, FP.PEAK_NEIGH_F, FP.MIN_PEAK_DB, Sdb=Sdb, rise_db=3.0)
    if name == "whiten":
        W = Sdb - _local_mean(Sdb, 32)
        return _pick(W, FP.PEAK_NEIGH_T, FP.PEAK_NEIGH_F, FP.MIN_PEAK_DB, Sdb=Sdb, rise_db=6.0)
    raise ValueError(name)


def hashes(y, name="base"):
    """[(hash32, anchor_frame)], paired as audio.fingerprints pairs them."""
    if name == "base":
        return FP.hashes(y)
    pk = peaks(y, name)
    out = []
    for i, (t1, f1) in enumerate(pk):
        made = 0
        for t2, f2 in pk[i + 1:]:
            dt = t2 - t1
            if dt < FP.DT_MIN:
                continue
            if dt > FP.DT_MAX:
                break
            h = (int(f1 // FP.FREQ_QUANT) & 0x3FF) << 22 | (int(f2 // FP.FREQ_QUANT) & 0x3FF) << 12 | (int(dt // FP.DT_QUANT) & 0xFFF)
            out.append((h, t1))
            made += 1
            if made >= FP.FANOUT:
                break
    return out
