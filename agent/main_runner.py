import os

from agent.agent57 import Agent57


def run_gym_agent57(
        enable_train,
        env,
        env_name,
        kwargs,
        nb_trains=999_999_999,
        nb_time=999_999_999,
        logger_type=LoggerType.TIME,
        log_interval=0,
        test_env=None,
        test_episodes=10,
        is_load_weights=False,
        checkpoint_interval=0,
        movie_save=False,
    ):
    base_dir = os.path.join("tmp_{}".format(env_name))
    os.makedirs(base_dir, exist_ok=True)
    print("nb_time  : {:.2f}m".format(nb_time/60))
    print("nb_trains: {}".format(nb_trains))
    weight_file = os.path.join(base_dir, "{}_weight.h5".format(env_name))

    manager = Agent57(**kwargs)

    if test_env is None:
        test_actor = None
    else:
        test_actor = kwargs["actors"][0]
    # log = DisTrainLogger(
    #     logger_type,
    #     interval=log_interval,
    #     savedir=base_dir,
    #     test_actor=test_actor,
    #     test_env=test_env,
    #     test_episodes=test_episodes,
    #     test_save_max_reward_file=os.path.join(base_dir, 'max_{step:02d}_{reward}.h5')
    # )

    if enable_train:
        print("--- start ---")
        print("'Ctrl + C' is stop.")
        save_manager = DisSaveManager(
            save_dirpath=base_dir,
            is_load=is_load_weights,
            save_memory=False,
            checkpoint=(checkpoint_interval>0),
            checkpoint_interval=checkpoint_interval,
            verbose=0
        )

        manager.train(nb_trains, nb_time, callbacks=[save_manager, log])

    # plt
    log.drawGraph("train")

    # View training results
    agent = manager.createTestAgent(kwargs["actors"][0], "tmp_{}/last/learner.dat".format(env_name))
    if agent is None:
        return
    agent.test(env, nb_episodes=5, visualize=True)

    # For video storage
    if movie_save:
        movie = MovieLogger()
        callbacks = [movie]
        if kwargs["input_type"] != InputType.VALUES:
            conv = ConvLayerView(agent)
            callbacks.append(conv)
        agent.test(env, nb_episodes=1, visualize=False, callbacks=callbacks)
        movie.save(gifname="tmp/{}_1.gif".format(env_name), fps=30)
        if kwargs["input_type"] != InputType.VALUES:
            conv.save(grad_cam_layers=[], add_adv_layer=True, add_val_layer=True,
                end_frame=200, gifname="tmp/{}_2.gif".format(env_name), fps=10)

    env.close()