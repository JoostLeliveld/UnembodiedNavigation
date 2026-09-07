def _predict_batch(self, images_bgr: list[np.ndarray]):
    """One inference cycle over the whole timestamp batch, in camera order.

        The cycle may be split into consecutive chunks, because activation memory --
        not compute -- is what limits this machine. Measured on the 4 GiB P2000 with
        Gazebo rendering five 1280x720 oblique cameras (which holds about 1.9 GiB of
        the card), reserved GPU memory for one cycle of five images is:

            imgsz 640: 708 MiB whole    | imgsz 960: 2558 MiB whole -> OOM in a live run
            imgsz 960 in chunks of two: 1418 MiB, which fits
            imgsz 1280: does not fit at any chunk size beside the simulator

        Chunking changes nothing observable: the batch is still one strict
        same-timestamp set, the results are still assembled in CAMERA_ORDER, and the
        reported inference time is still the wall time of the complete cycle.
        """
    if not 1 <= len(images_bgr) <= len(CAMERA_ORDER):
        raise BatchContractError(f'predict requires between 1 and {len(CAMERA_ORDER)} images')
    chunk = self.inference_chunk if self.inference_chunk > 0 else len(images_bgr)
    results: list[Any] = []
    for start in range(0, len(images_bgr), chunk):
        group = list(images_bgr[start:start + chunk])
        kwargs = {'source': group, 'imgsz': self.image_size, 'conf': self.predict_conf_floor, 'iou': self.iou_threshold, 'batch': len(group), 'stream': False, 'verbose': False}
        if self.device:
            kwargs['device'] = self.device
        cycle = getattr(self, '_active_cycle', None)
        started_wall = time.perf_counter()
        started_stamp = self._clock_s() if cycle is not None else None
        try:
            part = self.model.predict(**kwargs)
            if part is None:
                raise BatchContractError('batch inference returned no result sequence')
            results.extend(validate_batch_results(part, len(group)))
        except Exception as exc:
            if cycle is not None:
                self._publish_batch_outcome(dict(source_batch_id=cycle[0], invocation_id=f'{cycle[0]}/chunk/{start // chunk}', camera_ids=cycle[1][start:start + len(group)], status='inference_error', inference_start_stamp_s=started_stamp, reason=str(exc)))
            raise
        if cycle is not None:
            self._publish_batch_outcome(dict(source_batch_id=cycle[0], invocation_id=f'{cycle[0]}/chunk/{start // chunk}', camera_ids=cycle[1][start:start + len(group)], status='inference_completed', inference_start_stamp_s=started_stamp, inference_finish_stamp_s=self._clock_s(), inference_wall_ms=(time.perf_counter() - started_wall) * 1000.0))
    if len(results) != len(images_bgr):
        raise BatchContractError(f'chunked inference returned {len(results)} results for {len(images_bgr)} images')
    return results
