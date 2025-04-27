class Trader:
    def __init__(self, minute, max_len, pred_len, num, v_ss, rik, son, force=False):
        self.max_len = max_len
        self.pred_len = pred_len
        self.minute = minute
        self.price_list = []
        self.position = {'side': 0, 'count': 0, 'price': 0, 'std': 0}
        self.sort_range = []
        self.force = force

        self.params = {'max_length': max_len,
                       'predict_length': pred_len,
                       'number_of_sample': num,
                       'similar_value': v_ss,
                       'rikaku': rik,
                       'sonkiri': son}

        pickle_path_x = 'histdata/pk_{}_{}_{}_{}_data_x.pickle'.format(
            minute, max_len, pred_len, num)
        pickle_path_y = 'histdata/pk_{}_{}_{}_{}_data_y.pickle'.format(
            minute, max_len, pred_len, num)
        pickle_path_z = 'histdata/pk_{}_{}_{}_{}_data_z.pickle'.format(
            minute, max_len, pred_len, num)

        if not os.path.exists(pickle_path_x):
            hist_data = []
            with open('histdata/btcjpy_{}m.csv'.format(minute)) as f:
                reader = csv.reader(f)
                for row in reader:
                    hist_data.append(row[-1])
            hist_data = np.array(hist_data[1:], dtype='float32')

            temp = [np.roll(hist_data, -i, axis=0) for i in range(max_len)]
            data_x = np.stack(temp, axis=1)[:-(max_len - 1)][:-pred_len]

            mx = np.max(data_x, axis=1).reshape(len(data_x), 1)
            mn = np.min(data_x, axis=1).reshape(len(data_x), 1)
            norm = (data_x - mn) / (mx - mn)

            self.data_x = norm[:-pred_len]
            self.data_x = self.data_x.astype('float16')
            with open(pickle_path_x, 'wb') as web:
                pickle.dump(self.data_x, web)

            temp = [np.roll(hist_data[max_len:], -i, axis=0)
                    for i in range(pred_len)]
            data_z = np.stack(temp, axis=1)[:-(pred_len - 1)]

            mx = np.max(data_x, axis=1).reshape(len(data_z), 1)
            mn = np.min(data_x, axis=1).reshape(len(data_z), 1)
            norm = (data_z - mn) / (mx - mn)

            self.data_z = norm[:len(self.data_x)]
            self.data_z = self.data_z.astype('float16')
            with open(pickle_path_z, 'wb') as web:
                pickle.dump(self.data_z, web)

            a = self.data_z[:, -1] - self.data_z[:, 0]
            b = np.max(self.data_z, axis=1) - np.min(self.data_z, axis=1)
            self.data_y = a/b
            self.data_y = self.data_y.astype('float16')
            with open(pickle_path_y, 'wb') as web:
                pickle.dump(self.data_y, web)
        else:
            with open(pickle_path_x, 'rb') as f:
                self.data_x = pickle.load(f)
            with open(pickle_path_y, 'rb') as f:
                self.data_y = pickle.load(f)
            with open(pickle_path_z, 'rb') as f:
                self.data_z = pickle.load(f)

    def reset(self):
        self.price_list = []
        self.position = {'side': 0, 'count': 0, 'price': 0, 'std': 0}

    def ret_indication(self):
        price_np = np.array(self.price_list)
        mx, mn = np.max(price_np), np.min(price_np)
        norm = (price_np - mn) / (mx - mn)

        temp = np.sum(np.square(self.data_x - norm), axis=1)
        temp[np.isnan(temp)] = 0
        temp = 1 / (1 + temp)
        self.sort_range = np.argsort(temp)[::-1]

        differ = temp[self.sort_range][:self.params['number_of_sample']]
        indication = self.data_y[self.sort_range][:self.params['number_of_sample']]
        indicator = float(np.average(differ * indication))

        return indicator

    def refresh_position(self, price, count):
        try:
            if int(count/self.minute) == count/self.minute:
                if price:
                    self.price_list.append(price)
                if self.params['max_length'] <= len(self.price_list):
                    self.price_list = self.price_list[-self.params['max_length']:]

                    if not self.position['side'] == 0:
                        self.position['count'] -= 1

                        if self.position['count'] == 0:
                            self.position = {'side': 0,
                                             'count': 0, 'price': 0, 'std': 0}
                        else:
                            rik_son = (
                                price - self.position['price'])*self.position['side']/self.position['std']
                            if rik_son > self.params['rikaku'] or rik_son < -self.params['sonkiri']:
                                self.position = {
                                    'side': 0, 'count': 0, 'price': 0, 'std': 0}

                    if self.position['side'] == 0 or self.force:
                        indicator = self.ret_indication()
                        print(indicator)
                        if abs(indicator) > self.params['similar_value']:
                            std = np.std(np.array(self.price_list))
                            price = self.price_list[-1] if count == 0 else price
                            if indicator > 0:
                                self.position = {
                                    'side': 1, 'count': self.pred_len, 'price': price, 'std': std}
                            else:
                                self.position = {
                                    'side': -1, 'count': self.pred_len, 'price': price, 'std': std}

            print(self.position)
        except Exception as e:
            print('refresh_position')
            print(e)

    def return_similar(self):
        try:
            if not len(self.sort_range) == 0:
                mx = float(max(self.price_list))*10
                mn = float(min(self.price_list))*10
                temp_data_z = np.concatenate(
                    [self.data_x, self.data_z], axis=1)
                norm = temp_data_z[self.sort_range][:25]

                return (norm*(mx-mn) + mn)/10
            else:
                return []
        except Exception as e:
            print(e)


class MainScreen:
    def __init__(self, exchange_function, amount,
                 leverage=True,
                 emergency_stop_ratio=0.2):

        self.count, self.emergency = 0, False

        self.leverage = leverage
        self.emergency_stop_ratio = emergency_stop_ratio
        self.exchange_function = exchange_function

        self.data_list = [Trader(1, 90, 30, 1000, 0.06, 2, 2, force=True),
                          Trader(1, 90, 60, 1000, 0.07, 8, 1),
                          Trader(1, 60, 30, 1000, 0.09, 100, 6),
                          Trader(1, 60, 60, 500, 0.09, 2, 1.5)]

        self.amount = amount

    def init_status(self):
        self.count = 0
        self.exchange_function.cancel_all()
        self.exchange_function.zero_position(emergency=True)

    def run(self):
        try:
            self.exchange_function.cancel_all()

            if not self.emergency:
                self.count += 1
                asset = self.exchange_function.get_asset()

                if not asset == 'E':
                    price = self.exchange_function.get_price()

                    if not price == 'E':
                        for i in self.data_list:
                            i.refresh_position(price, self.count)
                        next_position = round(
                            sum([i.position['side'] for i in self.data_list]) * self.amount, 4)
                        now_position = self.exchange_function.get_position()

                        if not now_position == 'E':
                            self.exchange_function.make_position(
                                now_position, next_position)

            else:
                self.exchange_function.zero_position(emergency=True)
        except Exception as e:
            print(e)
