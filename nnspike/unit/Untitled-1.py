        # 4. 青ライン検出後、右モーターBの移動距離600未満の間は左エッジトレース、600到達でPAUSE（状態リセット）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_limit_reached = abs(current_pos - position_start) >= 400
            if position_limit_reached:
                phase.next_phase()
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.HEAD_GOAL

        # 4. 青ライン検出後、右モーターBの移動距離600未満の間は左エッジトレース、600到達でPAUSE（状態リセット）
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_limit_reached = abs(current_pos - position_start) >= 200
            if position_limit_reached:
                phase.next_phase()
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, None, Mode.HEAD_GOAL

        # 5. 600到達で状態リセットしPAUSE
        if phase.get_phase() == 6:
            self.reset_action()
            return None, None, Mode.PAUSE